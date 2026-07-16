from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectConfig:
    class_a: str = os.getenv("CLASS_A", "cat")
    class_b: str = os.getenv("CLASS_B", "dog")
    data_root: Path = Path(os.getenv("DATA_ROOT", "data"))
    img_size: tuple[int, int] = (int(os.getenv("IMG_SIZE", "128")), int(os.getenv("IMG_SIZE", "128")))
    img_size_tl: tuple[int, int] = (160, 160)
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    seed: int = int(os.getenv("SEED", "42"))
    outputs_dir: Path = Path("outputs")
    models_dir: Path = Path("models")
    exports_dir: Path = Path("exports")
    logs_dir: Path = Path("logs")


CONFIG = ProjectConfig()
