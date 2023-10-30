from __future__ import annotations

import random
import wandb

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam

from multiscale_operator.data import get_catalog
from multiscale_operator.data.datasets import MSODataModule, OverfitDataModule
from multiscale_operator.viz.visualization import Visualization


class MSOModule(pl.LightningModule):
    def __init__(self, operator: torch.nn.Module, cfg: DictConfig | dict):
        super().__init__()
        self.cfg = cfg
        self.operator = operator
        self.loss = torch.nn.MSELoss()

    def forward(self, batch):
        return self.operator(batch)

    def training_step(self, batch, batch_idx):
        yhat = self.operator(batch)
        loss = self.loss(yhat, batch.y)

        self.log("train_loss", loss, batch_size=batch.num_graphs)
        return loss

    def metrics(self, batch, yhat):
        mean_err = []
        nrm = []

        for idx in range(batch.num_graphs):
            sample = batch.get_example(idx)
            yhat_sample = yhat[batch.batch == idx]

            mean_err.append(torch.sqrt(torch.mean((yhat_sample - sample.y) ** 2)))
            nrm.append(torch.sqrt(torch.mean(sample.y**2)))

        mean_err = torch.stack(mean_err)
        nrm = torch.stack(nrm)

        err_RMSE = torch.mean(mean_err)
        err_nRMSE = torch.mean(mean_err / nrm)
        return {"RMSE": err_RMSE, "nRMSE": err_nRMSE}

    def validation_step(self, batch, batch_idx):
        yhat = self.operator(batch)
        loss = self.loss(yhat, batch.y)

        if batch_idx % self.cfg.plot_every_n_batches == 0:
            sample = batch.get_example(0)
            yhat_val = yhat[batch.batch == 0]

            # make sure we really only plot one sample
            vmin = min(sample.y[:, 0].min(), yhat_val[:, 0].min())
            vmax = max(sample.y[:, 0].max(), yhat_val[:, 0].max())

            # log an image
            im0 = (
                Visualization(
                    sample.pos[:, 0].detach().cpu().numpy(),
                    sample.pos[:, 1].detach().cpu().numpy(),
                    axis_off=True,
                )
                .plot_on_grid(sample.x[:, 0].detach().cpu().numpy(), scatter=False)
                .to_numpy()
            )

            im1 = (
                Visualization(
                    sample.pos[:, 0].detach().cpu().numpy(),
                    sample.pos[:, 1].detach().cpu().numpy(),
                    axis_off=True,
                )
                .plot_on_grid(
                    sample.y[:, 0].detach().cpu().numpy(), scatter=False, vmax=vmax, vmin=vmin
                )
                .to_numpy()
            )
            im2 = (
                Visualization(
                    sample.pos[:, 0].detach().cpu().numpy(),
                    sample.pos[:, 1].detach().cpu().numpy(),
                    axis_off=True,
                )
                .plot_on_grid(
                    yhat_val[:, 0].detach().cpu().numpy(), scatter=False, vmax=vmax, vmin=vmin
                )
                .to_numpy()
            )
            wandb.log(
                {
                    "output": wandb.Image(np.concatenate([im0, im2, im1])),
                }
            )

        metrics = self.metrics(batch, yhat)
        metrics = {f"val/{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch, batch_idx):
        metrics = self.metrics(batch, self.operator(batch))
        metrics = {f"test/{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, prog_bar=True, logger=True, batch_size=batch.num_graphs)

        loss = self.loss(self.operator(batch), batch.y)
        self.log("test_mse", loss, prog_bar=True, logger=True, batch_size=batch.num_graphs)

        return metrics

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.cfg.train_cfg.learning_rate)
        T_max = self.cfg.train_cfg.num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

        return [optimizer], [scheduler]


@hydra.main(config_path="../../configs", version_base=None)
def main(cfg: DictConfig | dict):
    if cfg.test_overfitting:
        cfg.train_cfg.log_every_n_steps = 1
        cfg.train_cfg.batch_size = 1
    if cfg.train_cfg.seed is None:
        cfg.train_cfg.seed = random.randint(0, 10000)
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.train_cfg.seed)
    random.seed(cfg.train_cfg.seed)
    np.random.seed(cfg.train_cfg.seed)

    with wandb.init(**cfg.wandb):
        # init catalog
        catalog = get_catalog()

        # load data
        driver = catalog[cfg.dataset.catalog_key].get_driver()
        data_module = MSODataModule(cfg, driver=driver)
        sample = data_module.get_sample()

        # create model
        operator = hydra.utils.instantiate(cfg.model_cfg.operator, cfg)
        operator.init_shapes(sample)
        module = MSOModule(operator, cfg)

        # training boilerplate
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=3, save_last=True)
        learning_rate_monitor = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(
            logger=WandbLogger(log_model=True),
            callbacks=[checkpoint_callback, learning_rate_monitor],
            max_epochs=cfg.train_cfg.num_epochs,
            log_every_n_steps=cfg.train_cfg.log_every_n_steps,
            check_val_every_n_epoch=cfg.train_cfg.check_val_every_n_epoch,
            accumulate_grad_batches=cfg.train_cfg.accumulate_grad_batches,
        )

        if cfg.test_overfitting:
            data_module = OverfitDataModule(sample)

        trainer.fit(module, datamodule=data_module)
        trainer.test(ckpt_path="best", datamodule=data_module)


if __name__ == "__main__":
    main()
