"""Configuration loading shared by every script."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_root"] = os.path.dirname(os.path.abspath(path))
    return cfg


def abspath(cfg, rel):
    return rel if os.path.isabs(rel) else os.path.join(cfg.get("_root", ROOT), rel)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
