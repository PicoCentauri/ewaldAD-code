import os
from datetime import datetime
import argparse
from ruamel.yaml import YAML

import torch
import lightning.pytorch as pl

from bpps_lode_model import BPPSLodeModel
from lightning_model import LitModel
from atomistic_dataset import LitDataModule

DATE_FORMAT = "%d-%m-%Y--%H:%M:%S"

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def load_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", type=str)
    args = parser.parse_args()
    with open(args.parameters, "r") as f:
        yaml = YAML(typ="safe", pure=True)
        parameters = yaml.load(f)
    return parameters


def prepare_datamodule(parameters: dict) -> LitDataModule:
    datamodule = LitDataModule(**parameters["datamodule"])
    return datamodule


def prepare_model(datamodule: LitDataModule, parameters: dict) -> BPPSLodeModel:
    compositions_weights = datamodule.prepare_compositions_weights()
    model = BPPSLodeModel(
        unique_elements=datamodule.unique_elements, **parameters["model"]
    )
    model.compositions_weights = compositions_weights
    print(model)
    return model


def prepare_litmodel(model: BPPSLodeModel, parameters: dict) -> LitModel:
    restart = parameters["litmodel"].pop("restart")
    if restart:
        litmodel = LitModel.load_from_checkpoint(
            restart, model=model, **parameters["litmodel"]
        )
    else:
        litmodel = LitModel(model=model, **parameters["litmodel"])
    return litmodel


def prepare_logger(parameters: dict) -> pl.loggers.WandbLogger:
    logname = (
        parameters["logging"].pop("name") + f"_{datetime.now().strftime(DATE_FORMAT)}"
    )
    logdir = parameters["logging"].pop("save_dir")
    os.makedirs(logdir, exist_ok=True)
    logger = pl.loggers.WandbLogger(
        name=logname, save_dir=logdir, **parameters["logging"]
    )
    logger.experiment.config.update(parameters)
    return logger


def prepare_trainer(logger: pl.loggers.WandbLogger, parameters: dict) -> pl.Trainer:
    checkpoint_callback = parameters["trainer"].pop("checkpoint_callback")
    callbacks = [
        pl.callbacks.ModelCheckpoint(**checkpoint_callback),
    ]
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **parameters["trainer"],
        inference_mode=False,
    )
    return trainer


def main():
    parameters = load_parameters()
    datamodule = prepare_datamodule(parameters)
    model = prepare_model(datamodule, parameters)
    litmodel = prepare_litmodel(model, parameters)
    logger = prepare_logger(parameters)
    trainer = prepare_trainer(logger, parameters)

    trainer.fit(litmodel, datamodule=datamodule)
    if datamodule.test_dataset:
        trainer.test(litmodel, datamodule=datamodule)


if __name__ == "__main__":
    main()
