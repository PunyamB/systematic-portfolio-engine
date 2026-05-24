# utils/config_loader.py
# Loads settings.yaml and .env once at startup.
# Every module imports get_config() instead of reading files directly.

import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env credentials into environment variables
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_config = None

def get_config() -> dict:
    """
    Returns the full settings.yaml as a dictionary.
    Cached after first load — file is only read once per session.
    """
    global _config
    if _config is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def get_env(key: str) -> str:
    """
    Returns a credential from .env by key.
    Raises clearly if the key is missing so nothing silently fails.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value