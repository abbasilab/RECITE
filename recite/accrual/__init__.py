"""
Accrual pipeline: screen papers for EC directives and impact quantification,
match to trials, compute scalar gains, and summarize.
"""

from recite.accrual.db import (
    get_paper_answers,
    get_trial_metadata_enrollment,
    get_top_papers,
    init_accrual_db,
    insert_paper_answer,
    insert_paper_trial_gain,
)
from recite.accrual.parsing import parse_directives_response, parse_impact_response
from recite.accrual.prompts import load_accrual_prompts

__all__ = [
    "get_paper_answers",
    "get_trial_metadata_enrollment",
    "get_top_papers",
    "init_accrual_db",
    "insert_paper_answer",
    "insert_paper_trial_gain",
    "load_accrual_prompts",
    "parse_directives_response",
    "parse_impact_response",
]
