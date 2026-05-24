"""
data_manager.py — Persistent storage for TradingBotV1
Uses Railway volume mount if available, falls back to local data/ folder.
"""
import os
import json
from datetime import datetime

# Railway persistent volume path
# Set RAILWAY_VOLUME_MOUNT_PATH in Railway environment variables
# Default fallback to local data/

VOLUME_PATH = os.environ.get(
    "RAILWAY_VOLUME_MOUNT_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data"))


def get_path(filename: str) -> str:
    """
    Get full path for a data file.
    Uses Railway volume if available, otherwise local data/ folder.
    """
    os.makedirs(VOLUME_PATH, exist_ok=True)
    return os.path.join(VOLUME_PATH, filename)


def save_json(filename: str, data: object) -> bool:
    """Save data to persistent storage."""
    try:
        path = get_path(filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"[DataManager] Save error {filename}: {e}")
        return False


def load_json(filename: str, default=None) -> object:
    """Load data from persistent storage."""
    try:
        path = get_path(filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default
    except Exception as e:
        print(f"[DataManager] Load error {filename}: {e}")
        return default


def list_files(prefix="") -> list:
    """List all files in storage."""
    try:
        files = []
        for f in os.listdir(VOLUME_PATH):
            if f.startswith(prefix):
                files.append(f)
        return files
    except Exception:
        return []


print(f"[DataManager] Storage path: {VOLUME_PATH}")
