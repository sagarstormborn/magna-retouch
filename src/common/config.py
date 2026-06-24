from pathlib import Path
import yaml


def load_config(path: str | Path = "configs/pipeline.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
