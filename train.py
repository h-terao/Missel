import hydra
from omegaconf import DictConfig
import tensorflow as tf
import dotenv

try:
    import comet_ml  # noqa: F401
except ImportError:
    pass

dotenv.load_dotenv(override=True)
tf.config.experimental.set_visible_devices([], "GPU")


@hydra.main(config_path="config/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train_pipeline import train
    from src.utils import utils

    utils.setup_main(config)
    return train(config)


if __name__ == "__main__":
    main()
