"""Load config.yaml into a plain nested dict."""
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    with open(Path(path)) as f:
        return yaml.safe_load(f)


def parse_source(source):
    """Webcam index as int ('1' or 1), anything else (file path / URL) as str."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    return int(s) if s.isdigit() else s
