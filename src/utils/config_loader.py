"""
Configuration Loader Utility
Loads and provides access to project configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """
    Singleton-style config loader that reads configs/config.yaml
    and provides dot-notation-style access.
    """

    _instance: Optional["ConfigLoader"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = "configs/config.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: str) -> None:
        """Load YAML configuration file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Retrieve a nested config value by key path.
        Example: config.get("models", "random_forest", "n_estimators")
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def as_dict(self) -> Dict[str, Any]:
        return self._config.copy()


def load_config(config_path: str = "configs/config.yaml") -> ConfigLoader:
    """Convenience function to get the config loader instance."""
    # Reset singleton for fresh load when path differs
    ConfigLoader._instance = None
    return ConfigLoader(config_path)
