"""A training pipeline."""
from __future__ import annotations

import jax.random as jr
from flax.traverse_util import flatten_dict
from hazuchi import Trainer
from hazuchi.callbacks.callback import Callback
import rich
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import utils


logger = utils.get_logger(__name__)


def train(config: DictConfig):
    logger.info(f"Seed: {config.seed}")
    rng = jr.PRNGKey(config.seed)

    logger.info(f"Instantiating dataset <{config.dataset._target_}>")
    dataset = hydra.utils.instantiate(config.dataset)
    train_loader = dataset.train_loader()
    test_loader = dataset.test_loader()
    batch = next(iter(train_loader))

    logger.info(f"Instantiating model <{config.model._target_}>")
    train_steps = config.max_epochs * config.epoch_length
    learner = hydra.utils.instantiate(
        config.learner,
        data_meta=dataset.data_meta,
        train_steps=train_steps,
        tx={"train_steps": train_steps},
    )

    callbacks: dict[str, Callback] = {}
    for key, conf in config.callbacks.items():
        if conf is not None:
            logger.info(f"Instantiating callback <{conf._target_}>")
            if (
                conf._target_ == "hazuchi.callbacks.PrintMetrics"
                and conf.get("entities", None) is None
            ):
                entries = ["epoch"]
                entries += [f"train/{x}" for x in learner.default_entries]
                entries += [
                    "val/loss",
                    "val/acc1",
                    "val/acc5",
                    "val/EMA/acc1",
                    "val/best_acc1",
                    "elapsed_time",
                ]
                conf["entries"] = entries
            callbacks[key] = hydra.utils.instantiate(conf)

    logger.info("Instantiating trainer")
    trainer = Trainer(
        out_dir="./",
        max_epochs=config.max_epochs,
        train_fn=learner.train_fn,
        val_fn=learner.test_fn,
        callbacks=callbacks,
        prefetch=config.prefetch,
    )

    logger.info("Initializing train_state!")
    train_state, model_info = learner.init_fn(rng, batch)

    logger.info("Logging hyperparameters!")
    hparams = flatten_dict(OmegaConf.to_container(config, resolve=True), sep="/")
    hparams = dict(hparams, **model_info)
    trainer.log_hyperparams(hparams)
    rich.print(hparams)

    logger.info("Starting training!")
    train_state = trainer.fit(
        train_state,
        train_loader,
        test_loader,
        train_len=config.epoch_length,
        val_len=dataset.data_meta["test_steps_per_epoch"],
    )

    logger.info("Finalizing!")
    trainer.finalize()
