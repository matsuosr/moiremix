import logging
import pathlib
from dataclasses import dataclass, field
from typing import Final, Tuple, cast, List, Any

import hydra
import torch
import torchvision
import torchvision.models as models
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

import fhmap
import fhmap.schema as schema
from fhmap.factory.dataset import ImagenetDataModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class EvalFhmapConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"dataset": "imagenet"},
        {"env": "default"},
        "_self_"
    ])
    env: schema.EnvConfig = schema.DefaultEnvConfig  # type: ignore
    dataset: schema.DatasetConfig = schema.ImagenetConfig  # type: ignore
    batch_size: int = 128
    eps: float = 15.7
    ignore_edge_size: int = 96
    num_samples: int = 1000
    topk: Tuple = (1, 5)
    weightpath: str = MISSING

cs = ConfigStore.instance()
cs.store(name="eval_fhmap_vit", node=EvalFhmapConfig)
cs.store(group="dataset", name="imagenet", node=schema.ImagenetConfig)
cs.store(group="env", name="default", node=schema.DefaultEnvConfig)

@hydra.main(config_path=None, config_name="eval_fhmap_vit", version_base="1.1")
def eval_fhmap(cfg: EvalFhmapConfig) -> None:
    OmegaConf.set_readonly(cfg, True)
    logger.info(OmegaConf.to_yaml(cfg))

    device: Final = cfg.env.device
    cwd: Final[pathlib.Path] = pathlib.Path(hydra.utils.get_original_cwd())
    weightpath: Final[pathlib.Path] = pathlib.Path(cfg.weightpath)

    # 1. Setup datamodule
    root: Final[pathlib.Path] = cwd / "data"
    datamodule = ImagenetDataModule(
        batch_size=cfg.batch_size,
        num_workers=cfg.env.num_workers,
        root=root
    )
    datamodule.prepare_data()
    datamodule.setup()

    # --- サンプリング枚数を制限する工夫（AttributeError対策版） ---
    test_dataset = datamodule.test_dataset
    if cfg.num_samples > 0:
        logger.info(f"Subsampling dataset to {cfg.num_samples} images while preserving attributes.")
        # Subsetを使わず、内部のsamplesリストを直接切り詰める（ImageFolderの場合）
        if hasattr(test_dataset, 'samples'):
            test_dataset.samples = test_dataset.samples[:cfg.num_samples]
            test_dataset.targets = test_dataset.targets[:cfg.num_samples]
        # その他のDatasetタイプの場合の予備策
        elif hasattr(test_dataset, 'imgs'):
            test_dataset.imgs = test_dataset.imgs[:cfg.num_samples]

    # 2. Setup ViT model
    arch = models.vit_b_16(weights=None)
    arch.heads.head = torch.nn.Linear(arch.heads.head.in_features, datamodule.num_classes)
    
    checkpoint = torch.load(weightpath, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    arch.load_state_dict(clean_state_dict, strict=True)
    
    arch = arch.to(device)
    arch.eval()

    # 3. Fourier Heat Map Evaluation
    fhmap.eval_fourier_heatmap(
        datamodule.input_size,
        cfg.ignore_edge_size, 
        cfg.eps,
        arch,
        test_dataset, # 属性を保持したまま枚数を減らしたオブジェクト
        cfg.batch_size,
        cast(torch.device, device),
        cfg.topk,
        pathlib.Path("."),
    )

if __name__ == "__main__":
    eval_fhmap()