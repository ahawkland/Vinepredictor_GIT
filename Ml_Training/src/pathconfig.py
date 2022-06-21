from dataclasses import dataclass, field
import typing as t
from typing import Any, Optional
from pathlib import Path


@dataclass()
class PathConfig:
    path: t.Optional[Path] = 'configs/model.yaml'
    base_path: t.Optional[Path] = Path(__file__).absolute().parent.parent
    configs: t.Optional[Path] = None

    def __post_init__(self):
        self.configs = self.base_path.joinpath("configs")
        self.path = self.base_path
        self.model_config = self.configs.joinpath("model.yaml")
        self.output = self.base_path.joinpath("output")
        self.data = self.base_path.joinpath("Data")


if __name__ == "__main__":
    path_repo = PathConfig()
    print(path_repo.model_config)
