from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import time

from multiscale_operator.data import get_catalog
from multiscale_operator.data.datasets import MSODataModule
from multiscale_operator.model.trainer import MSOModule
from multiscale_operator.viz.visualization import Visualization
from multiscale_operator.model.wandb_helper import init as wandb_init
from pytorch_lightning.loggers import WandbLogger


@hydra.main(config_path="../../configs", version_base=None)
def main(cfg: DictConfig | dict):
    cfg.train_cfg.num_workers = 0
    cfg.test_cfg.batch_size = 1
    print(OmegaConf.to_yaml(cfg))

    with wandb_init(**cfg.wandb):
        # init catalog
        catalog = get_catalog()

        # load data
        driver = catalog[cfg.dataset.catalog_key].get_driver()
        data_module = MSODataModule(cfg, driver=driver)
        sample = data_module.get_sample()

        # init op
        operator = hydra.utils.instantiate(cfg.model_cfg.operator, cfg)
        operator.init_shapes(sample)

        # load checkpoint
        api = wandb.Api()
        artifact = api.artifact(cfg.load_artifact, type="model")
        artifact_dir = artifact.download()
        print(f"Loaded artifact from {artifact_dir}")

        module = MSOModule.load_from_checkpoint(
            Path(artifact_dir) / "model.ckpt", map_location="cpu", operator=operator, cfg=cfg
        )

        trainer = pl.Trainer(logger=WandbLogger())

        start = time.time()
        trainer.test(model=module, datamodule=data_module)
        end = time.time()
        wandb.log({"test_time": end - start})


if __name__ == "__main__":
    main()
