from pathlib import Path

import argparse
import yaml


DEFAULT_CONFIG = {
    "data_cache": {
        "root": "./data"
    },
    "datasets": {
    }
}


def error() -> dict:
    """
    Prints the error message and returns the default configuration
    """
    print("[config] No valid configuration found. Falling back to default")
    return DEFAULT_CONFIG


def read_yaml_config(path: str | Path) -> dict:
    """
    Imports a YAML configuration
    """

    with open(path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None or not isinstance(config, dict) or len(config) == 0:
        return error()

    return config


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", 
        type=str,
        default="./config.yaml"
    )

    args = parser.parse_args()
    return read_yaml_config(args.config)


