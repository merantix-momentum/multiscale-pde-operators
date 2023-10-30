import torch
from squirrel.catalog import Catalog, CatalogKey, Source
from squirrel.framework.plugins.plugin_manager import register_driver
from torch_geometric.data import Data
from squirrel.driver import IterDriver
from squirrel.iterstream import Composable

from multiscale_operator.data.driver import (
    FsgMotorDriver,
    MessagepackHookDriver,
    PDEBenchDriver,
)


class MultiSourceDriver(IterDriver):
    name = "MultiSourceDriver"

    def __init__(self, key_train_source, key_val_source, key_test_source, **kwargs) -> None:
        super().__init__(**kwargs)
        catalog = get_catalog()
        self.train_source = catalog[key_train_source].get_driver()
        self.val_source = catalog[key_val_source].get_driver()
        self.test_source = catalog[key_test_source].get_driver()

    def get_iter(self, split, **kwargs) -> Composable:
        """
        Get an iterator over samples.
        """
        assert split in ["train", "val", "test"]
        if split == "train":
            return self.train_source.get_iter(train=True, **kwargs)
        elif split == "val":
            return self.val_source.get_iter(**kwargs)
        else:
            return self.test_source.get_iter(**kwargs)


def get_catalog():
    # register drivers
    for driver in DRIVERS:
        register_driver(driver)

    # init catalog
    catalog = Catalog()

    for key, source in SOURCES:
        catalog[key] = source

    return catalog


def extract_pos_hook(sample):
    return Data(
        x=torch.tensor(sample["data_x"][:, 2:]),
        y=torch.tensor(sample["data_y"][:, :1]),
        pos=torch.tensor(sample["data_x"][:, :2]),
        edge_index=torch.tensor(sample["edge_index"]),
    )


def to_pyg_hook(sample):
    return Data(
        x=torch.tensor(sample["x"]),
        y=torch.tensor(sample["y"]),
        pos=torch.tensor(sample["pos"]),
        edge_index=torch.tensor(sample["edge_index"]),
    )


DRIVERS = [PDEBenchDriver, MessagepackHookDriver, FsgMotorDriver, MultiSourceDriver]


SOURCES = [
    (
        CatalogKey("MotorDataset", 1),
        Source(
            driver_name="motor_driver",
            driver_kwargs={
                "base_dir": "datasets/motor/BM2",
            },
        ),
    ),
    (
        CatalogKey("DarcyDataset", 1),
        Source(
            driver_name="PDEBenchDriver",
            driver_kwargs={
                "filename": "datasets/darcy/pde_bench/2D_DarcyFlow_beta$BETA$_Train.hdf5",
                "beta": 1.0,
            },
        ),
    ),
    (
        CatalogKey("MagnetostaticsDataset", 1),
        Source(
            driver_name="MultiSourceDriver",
            driver_kwargs={
                "key_train_source": "GnnBvpMsTrainMa",
                "key_val_source": "GnnBvpMsValMa",
                "key_test_source": "GnnBvpMsTestShape",
            },
        ),
    ),
    (
        CatalogKey("MagnetostaticsSuperpositionsDataset", 1),
        Source(
            driver_name="MultiSourceDriver",
            driver_kwargs={
                "key_train_source": "GnnBvpMsTrainMa",
                "key_val_source": "GnnBvpMsValMa",
                "key_test_source": "GnnBvpMsTestSup",
            },
        ),
    ),
    (
        CatalogKey("GnnBvpMsTrainMa", 1),
        Source(
            driver_name="messagepack_hook",
            driver_kwargs={
                "url": "datasets/electromagnetics/gnn_bvp_solver/MagneticsRandomCurrentGenerator/norm_train_ma",
                "hook": extract_pos_hook,
            },
        ),
    ),
    (
        CatalogKey("GnnBvpMsValMa", 1),
        Source(
            driver_name="messagepack_hook",
            driver_kwargs={
                "url": "datasets/electromagnetics/gnn_bvp_solver/MagneticsRandomCurrentGenerator/norm_val_ma",
                "hook": extract_pos_hook,
            },
        ),
    ),
    (
        CatalogKey("GnnBvpMsTestShape", 1),
        Source(
            driver_name="messagepack_hook",
            driver_kwargs={
                "url": "datasets/electromagnetics/gnn_bvp_solver/MagneticsRandomCurrentGenerator/norm_test_shape",
                "hook": extract_pos_hook,
            },
        ),
    ),
    (
        CatalogKey("GnnBvpMsTestSup", 1),
        Source(
            driver_name="messagepack_hook",
            driver_kwargs={
                "url": "datasets/electromagnetics/gnn_bvp_solver/MagneticsRandomCurrentGenerator/norm_test_sup",
                "hook": extract_pos_hook,
            },
        ),
    ),
]
