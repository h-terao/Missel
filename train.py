import hydra
from omegaconf import DictConfig
import dotenv

try:
    import comet_ml  # noqa: F401
except ImportError:
    pass

dotenv.load_dotenv(override=True)


@hydra.main(config_path="config/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    import tensorflow as tf

    from src.train_pipeline import train
    from src.utils import utils

    tf.config.experimental.set_visible_devices([], "GPU")
    utils.setup_main(config)
    return train(config)


if __name__ == "__main__":
    main()
