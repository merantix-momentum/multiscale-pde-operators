from __future__ import annotations

import h5py
import hydra
import numpy as np
import pytorch_lightning as pl
import torch

from typing import Iterable
from omegaconf import DictConfig
from squirrel.driver import IterDriver
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.nn import radius_graph


class DarcyFlowDataset(Dataset):
    def __init__(
        self,
        filename,
        reduced_resolution=1,
        reduced_batch=1,
        split="train",
        test_ratio=0.1,
        val_ratio=0.1,
        num_samples_max=-1,
    ):
        assert split in ["train", "val", "test"]
        assert filename[-2:] != "h5", "HDF5 data is assumed!!"

        with h5py.File(filename, "r") as f:
            keys = list(f.keys())
            keys.sort()

            _data = np.array(f["tensor"], dtype=np.float32)
            _data = _data[::reduced_batch, :, ::reduced_resolution, ::reduced_resolution]
            _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
            self.data = _data
            _data = np.array(f["nu"], dtype=np.float32)
            _data = _data[::reduced_batch, None, ::reduced_resolution, ::reduced_resolution]
            _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
            self.data = np.concatenate([_data, self.data], axis=-1)
            self.data = self.data[:, :, :, :, None]

            x = np.array(f["x-coordinate"], dtype=np.float32)
            y = np.array(f["y-coordinate"], dtype=np.float32)
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]

        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        val_idx = test_idx + int(num_samples_max * val_ratio)
        if split == "test":
            # keep test data the same like for the original paper
            self.data = self.data[:test_idx]
        elif split == "val":
            self.data = self.data[test_idx:val_idx]
        else:
            self.data = self.data[val_idx:]

        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data[idx]

        # hack to load darcy data correctly
        x = x[:, :, 0:1]
        y = y[:, :, 1:2]

        x = x.reshape(-1, x.shape[2])
        y = y.reshape(-1, y.shape[2])
        grid = self.grid.reshape(-1, self.grid.shape[2])

        return Data(
            x=x,
            y=y,
            pos=grid,
            # create edge index s.t. each node is connected to its 8 closest neighbors
            edge_index=radius_graph(grid, r=torch.abs(grid[0] - grid[1]).sum() * 1.5),
        )


class MSOIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, driver: IterDriver, split: str, transform=None) -> None:
        """Creates a PyTorch iterable dataset from a squirrel messagepack driver"""
        self.driver = driver
        self.split = split

        if isinstance(transform, DictConfig):
            self.transform = hydra.utils.instantiate(transform)
        else:
            self.transform = transform

    def __iter__(self):
        """Iterate dataset"""
        for sample in self.driver.get_iter(split=self.split):
            if self.transform is not None:
                yield self.transform(sample)
            else:
                yield sample


class MSODataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig | dict, driver: IterDriver | Iterable[Data] | IterableDataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset = MSOIterableDataset(driver, split="train", transform=cfg.data_transform)
        self.val_dataset = MSOIterableDataset(driver, split="val", transform=cfg.data_transform)
        self.test_dataset = MSOIterableDataset(driver, split="test", transform=cfg.data_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_cfg.batch_size,
            num_workers=self.cfg.train_cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train_cfg.batch_size,
            num_workers=self.cfg.train_cfg.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.test_cfg.batch_size,
            num_workers=self.cfg.train_cfg.num_workers,
        )

    def get_sample(self):
        for batch in self.test_dataloader():
            return batch


class OverfitDataModule(pl.LightningDataModule):
    def __init__(self, overfit_sample) -> None:
        super().__init__()
        self.overfit_sample = overfit_sample

    def train_dataloader(self):
        return DataLoader([self.overfit_sample], batch_size=1)

    def val_dataloader(self):
        return DataLoader([self.overfit_sample], batch_size=1)

    def test_dataloader(self):
        return DataLoader([self.overfit_sample], batch_size=1)
