from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch_geometric as tg
from squirrel.driver import IterDriver, MessagepackDriver
from squirrel.iterstream import Composable, IterableSource
from torch_geometric.transforms import Compose, Delaunay, FaceToEdge

from multiscale_operator.data.datasets import DarcyFlowDataset


class PDEBenchDriver(IterDriver):
    name = "PDEBenchDriver"

    def __init__(self, filename, beta, **kwargs) -> None:
        super().__init__(**kwargs)
        self.filename = filename.replace("$BETA$", str(beta))

    def get_iter(self, split, **kwargs) -> Composable:
        """
        Get an iterator over samples.

        Args:
            shuffle_item_buffer (int): the size of the buffer used to shuffle samples after being fetched. Please note
                the memory footprint of samples
        """
        assert split in ["train", "val", "test"]
        return IterableSource(DarcyFlowDataset(self.filename, split=split))


class MessagepackHookDriver(IterDriver):
    name = "messagepack_hook"

    def __init__(self, url: str, hook: callable = None, **kwargs) -> None:
        self.hook = hook
        self.url = url

    def get_iter(self, split="", subfolder: str = None, train: bool = False, **kwargs) -> Composable:
        """Create iterstream."""
        subfolder = "" if subfolder is None else f"/{subfolder}"
        url = f"{self.url}{subfolder}/{split}"
        subdriver = MessagepackDriver(url)

        shuffle_key_buffer = 1
        shuffle_item_buffer = 1

        # shuffling is required only for training
        if split == "train" or train:
            shuffle_key_buffer = 100
            shuffle_item_buffer = 100

        return (
            IterableSource(subdriver.store.keys())
            .shuffle(size=shuffle_key_buffer)
            .split_by_worker_pytorch()
            .async_map(subdriver.store.get, max_workers=8)
            .flatten()
            .async_map(self.hook, max_workers=8)
            .shuffle(size=shuffle_item_buffer)
        )


class FsgMotorDriver(IterDriver):
    name = "motor_driver"

    def __init__(
        self,
        base_dir: str | Path,
        rotor_pos="pos1_",
        test_ratio=0.1,
        val_ratio=0.1,
        random_seed=42,
        **kwargs,
    ) -> None:
        """Initialize the motor driver."""

        super().__init__(**kwargs)
        self.rotor_pos = rotor_pos

        if isinstance(base_dir, str):
            base_dir = Path(base_dir)

        collect_txts = []

        for file in base_dir.rglob("*Bnorm_matinfo.txt"):
            # check if we use only a single rotor position (angle)
            if self.rotor_pos is None:
                collect_txts.append(file)
            elif self.rotor_pos in file.name:
                collect_txts.append(file)

        random.seed(random_seed)
        self.collect_txts_test = random.sample(collect_txts, int(len(collect_txts) * test_ratio))
        self.collect_txts_train = [f for f in collect_txts if f not in self.collect_txts_test]
        self.collect_txts_val = random.sample(self.collect_txts_train, int(len(collect_txts) * val_ratio))
        self.collect_txts_train = [f for f in self.collect_txts_train if f not in self.collect_txts_val]

    def map_f(self, f):
        txt_start_idx = 9
        transform = Compose([Delaunay(), FaceToEdge()])

        with open(f) as source:
            xs = []
            ys = []
            norms = []
            materials = []

            for line in source.readlines()[txt_start_idx:]:
                values = line.split()
                x, y, bnorm, material = values
                xs.append(float(x))
                ys.append(float(y))
                norms.append(float(bnorm))
                materials.append(float(material))

            xs = np.array(xs)
            ys = np.array(ys)
            norms = np.array(norms)
            materials = np.array(materials)

            data = tg.data.Data(
                pos=torch.tensor(np.stack([xs, ys], axis=1), dtype=torch.float32),
                x=torch.tensor(np.stack([materials], axis=1), dtype=torch.float32),
                y=torch.tensor(norms, dtype=torch.float32).reshape(-1, 1),
            )

            return transform(data)

    def get_iter(self, split, **kwargs) -> Composable:
        """Create iterstream."""
        # TODO: add shuffle_item_buffer or use map driver directly
        assert split in ["train", "val", "test"]

        if split == "train":
            collect_txts = self.collect_txts_train
        elif split == "val":
            collect_txts = self.collect_txts_val
        else:
            collect_txts = self.collect_txts_test

        return IterableSource(collect_txts).map(self.map_f)
