from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Project configuration and canonical data paths.

    Uses the repository root as two parents above this file (src/packagename/*).
    """

    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    SAMPLE_DIR: Path = DATA_DIR / "sample"

    SRC_DIR: Path = BASE_DIR / "src"
    CONFIGS_DIR: Path = BASE_DIR / "configs"
    SCRIPTS_DIR: Path = BASE_DIR / "scripts"


# convenience alias
config = Config()
