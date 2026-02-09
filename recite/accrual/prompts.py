"""Load accrual prompts from config/accrual_prompts.json."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from recite.utils.path_loader import get_project_root


def load_accrual_prompts(prompts_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load accrual_prompts.json. Path may be relative to project root."""
    if prompts_path is None:
        prompts_path = get_project_root() / "config" / "accrual_prompts.json"
    path = Path(prompts_path)
    if not path.is_absolute():
        path = get_project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Accrual prompts not found: {path}")
    with open(path) as f:
        return json.load(f)
