"""Utility to load paths from config/paths.yaml and data root from .env."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Load .env from project root so LOCAL_DB_DIR is available (used across machines)
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
    """Find project root by looking for config/paths.yaml or pyproject.toml"""
    current = Path(__file__).parent  # recite/utils/
    project_root = current.parent.parent  # project root (recite/utils/ -> recite/ -> project root)
    
    # Verify we're in the right place
    if not (project_root / "config" / "paths.yaml").exists():
        raise FileNotFoundError(
            f"Could not find config/paths.yaml. Searched from {Path(__file__)}"
        )
    
    return project_root


def get_data_root() -> Path:
    """
    Directory under which data subdirs live (dev, cluster1, cluster2, db_merge_staging, prod).
    Always project_root / "data". Use get_local_db_dir() for this machine's DB directory.
    """
    return get_project_root() / "data"


def get_local_db_dir() -> Path:
    """
    This machine's DB directory (e.g. data/dev or data/cluster1). Used for operating DB paths.
    Set .env LOCAL_DB_DIR (relative to project root or absolute). Default: data/dev.
    """
    root = get_project_root()
    raw = os.environ.get("LOCAL_DB_DIR", "data/dev")
    p = Path(raw)
    return p if p.is_absolute() else root / p


def resolve_path(p: Path, root: Optional[Path] = None) -> Path:
    """
    Resolve a path: if absolute return as-is; if starts with "data/", resolve under get_data_root();
    otherwise resolve under project root (or provided root).
    Use for DB paths and any path that may live under the configurable data directory.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    s = str(p).replace("\\", "/")
    if s.startswith("data/"):
        return get_data_root() / s[5:].lstrip("/")
    return (root if root is not None else get_project_root()) / p


def load_paths(config_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load paths from config/paths.yaml
    
    Args:
        config_file: Optional path to config file. If None, uses config/paths.yaml from project root.
    
    Returns:
        Dictionary containing all paths from the YAML file
    """
    if config_file is None:
        project_root = get_project_root()
        config_file = project_root / "config" / "paths.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file) as f:
        return yaml.safe_load(f)


def get_path(key_path: str, legacy: bool = False) -> Path:
    """Get a specific path from config.
    
    Args:
        key_path: Dot-separated path (e.g., 'config.criteria_labels')
        legacy: If True, prepends 'legacy_' to key_path sections
    
    Returns:
        Path object relative to project root
    
    Examples:
        >>> get_path('config.criteria_labels')
        Path('config/legacy/criteria-classification/labels.json')
        
        >>> get_path('config.criteria_labels', legacy=True)
        Path('config/legacy/criteria-classification/labels.json')  # Same, but from legacy_config section
        
        >>> get_path('legacy_scripts.criteria_classification', legacy=False)
        Path('scripts/legacy/criteria-classification')
    """
    paths = load_paths()
    project_root = get_project_root()
    
    if legacy:
        # Convert 'config.criteria_labels' -> 'legacy_config.criteria_labels'
        parts = key_path.split('.')
        parts[0] = f"legacy_{parts[0]}"
        key_path = '.'.join(parts)
    
    # Navigate nested dict
    value = paths
    for key in key_path.split('.'):
        if key not in value:
            raise KeyError(
                f"Path key '{key}' not found in config. "
                f"Available top-level keys: {list(paths.keys())}"
            )
        value = value[key]
    
    return project_root / value
