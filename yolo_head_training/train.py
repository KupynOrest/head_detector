from omegaconf import DictConfig
import hydra

from super_gradients import Trainer, init_trainer
from yolo_head import YoloHeadsLoss, KeypointsFailureRate, KeypointsNME, DAD3DHeadsDataset  # noqa

@hydra.main(config_path="configs", config_name="yolo_heads_m", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    Trainer.train_from_config(cfg)


def main() -> None:
    init_trainer()  # `init_trainer` needs to be called before `@hydra.main`
    _main()


if __name__ == "__main__":
    main()
