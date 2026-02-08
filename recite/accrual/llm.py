"""Call LLM for accrual paper Q&A. Supports local endpoint or UCSF Versa API."""

from typing import Optional

from recite.benchmark.evaluator import call_model_with_retry


def call_accrual_llm(
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    max_retries: int = 2,
    timeout: float = 120.0,
    ucsf_versa_model: Optional[str] = None,
) -> str:
    """Call LLM for accrual questions. If ucsf_versa_model is set, use UCSF Versa API; else use endpoint (OpenAI-compatible)."""
    if ucsf_versa_model is not None:
        from recite.llmapis import UCSFVersaAPI
        api = UCSFVersaAPI(model=ucsf_versa_model, system_prompt=system_prompt or "")
        return api(prompt, system_prompt, temperature=0)
    return call_model_with_retry(
        endpoint=endpoint,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        max_retries=max_retries,
        timeout=timeout,
    )
