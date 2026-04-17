"""Path resolution utilities."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        # Project root: this file is recite/utils/path_loader.py -> parent.parent
        _root = Path(__file__).resolve().parent.parent.parent
        load_dotenv(_root / ".env")
    except Exception:
        pass

_load_dotenv()


def get_project_root() -> Path:
    current = Path(__file__).parent
    project_root = current.parent.parent
    
    if not (project_root / "config" / "paths.yaml").exists():
        raise FileNotFoundError(
            f"Could not find config/paths.yaml. Searched from {Path(__file__)}"
        )
    
    return project_root


def get_data_root() -> Path:
    return get_project_root() / "data"


def get_local_db_dir() -> Path:
    root = get_project_root()
    raw = os.environ.get("LOCAL_DB_DIR", "data/dev")
    p = Path(raw)
    return p if p.is_absolute() else root / p


def resolve_path(p: Path, root: Optional[Path] = None) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    s = str(p).replace("\\", "/")
    if s.startswith("data/"):
        return get_data_root() / s[5:].lstrip("/")
    return (root if root is not None else get_project_root()) / p


def load_paths(config_file: Optional[Path] = None) -> Dict[str, Any]:
    if config_file is None:
        project_root = get_project_root()
        config_file = project_root / "config" / "paths.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file) as f:
        return yaml.safe_load(f)


def get_path(key_path: str, legacy: bool = False) -> Path:
    paths = load_paths()
    project_root = get_project_root()
    
    if legacy:
        parts = key_path.split('.')
        parts[0] = f"legacy_{parts[0]}"
        key_path = '.'.join(parts)
    
    value = paths
    for key in key_path.split('.'):
        if key not in value:
            raise KeyError(
                f"Path key '{key}' not found in config. "
                f"Available top-level keys: {list(paths.keys())}"
            )
        value = value[key]
    
    return project_root / value
