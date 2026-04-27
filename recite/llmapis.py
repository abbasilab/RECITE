"""LLM API wrappers for OpenAI and Azure OpenAI endpoints."""


import os
import json
import requests
import time
import re

from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger


load_dotenv()


class AbstractLLMAPI:
    def __init__(
            self,
            model: str,
            system_prompt: str,
            *args,
            **kwargs
            ) -> None:
        self.model = model
        logger.info(f"Initialized LLMAPI with model: {model}")
        
        self.system_prompt = system_prompt
        logger.debug(f"System prompt set for LLMAPI: {system_prompt[:60]}...")
        
        return

    def __call__(
            self,
            prompt: str,
            system_prompt: str,
            *args,
            **kwds
            ) -> str:
        logger.warning(
            "Abstract __call__ method invoked. This should be implemented in a subclass."
            )
        pass

    @staticmethod
    def robust_json_parse(
            llm_response: str
            ) -> dict | list:
        # Remove markdown/code block formatting
        llm_response = llm_response.strip()
        llm_response = re.sub(r"^```json\s*", "", llm_response)
        llm_response = re.sub(r"^```", "", llm_response)
        llm_response = re.sub(r"```$", "", llm_response)

        # Remove any leading/trailing whitespace or newlines
        llm_response = llm_response.strip()

        # Replace curly quotes with straight quotes
        llm_response = llm_response.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")

        # Remove trailing commas before closing braces/brackets
        llm_response = re.sub(r',(\s*[\]}])', r'\1', llm_response)

        return json.loads(llm_response)

    def tabulate_eligibility(
            self,
            prompt_template: str,
            *format_args,
            **format_kwargs
            ) -> str:
        prompt = prompt_template.format(*format_args, **format_kwargs)
        logger.debug("Calling LLM for tabulate_eligibility.")
        llm_response = self(prompt, self.system_prompt)
        return llm_response.strip()

    def tabulate_eligibility_json(
            self,
            prompt_template: str,
            *format_args,
            **format_kwargs
            ) -> dict:
        prompt = prompt_template.format(*format_args, **format_kwargs)
        logger.debug("Calling LLM for tabulate_eligibility_json.")
        llm_response = self(prompt, self.system_prompt)

        try:
            result = self.robust_json_parse(llm_response)
            if isinstance(result, dict) and 'headers' in result and 'entries' in result:
                return result
            else:
                logger.error("LLM response missing 'headers' or 'entries'.")
                return {'headers': [], 'entries': []}
        except Exception as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {'headers': [], 'entries': []}


class OpenAIAPI(AbstractLLMAPI):
    def __init__(
            self,
            model: str,
            system_prompt: str
            ) -> None:
        super().__init__(model, system_prompt)
        available_models = (
            "gpt-4.1-nano",
            "gpt-4.1-mini",
        )
        if model not in available_models:
            logger.error(f"Model '{model}' is not in the list of available models: {available_models}")
            raise ValueError(f"Model '{model}' is not in the list of available models: {available_models}")
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("OpenAIAPI client initialized.")
        return
    

    def __call__(
            self,
            prompt: str,
            system_prompt: str = None,
            temperature: float = 0
            ) -> str:
        if system_prompt is None:
            system_prompt = self.system_prompt

        logger.debug(f"Calling OpenAI API with model: {self.model}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature
            )
            output = response.choices[0].message.content.strip()
            logger.info("OpenAI API call successful.")
            logger.debug(f"OpenAI API output:\n{output}")
            return output
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return ""   
    
    def tabulate_eligibility(
            self,
            prompt_template: str,
            *format_args,
            **format_kwargs
            ) -> str:
        prompt = prompt_template.format(*format_args, **format_kwargs)
        llm_response = self(prompt, self.system_prompt)
        return llm_response.strip()

    def tabulate_eligibility_json(
            self,
            prompt_template: str,
            *format_args,
            **format_kwargs
            ) -> dict:
        prompt = prompt_template.format(*format_args, **format_kwargs)
        llm_response = self(prompt, self.system_prompt)

        try:
            result = self.robust_json_parse(llm_response)
            if isinstance(result, dict) and 'headers' in result and 'entries' in result:
                return result
            else:
                logger.error("Response missing 'headers' or 'entries'.")
                return {'headers': [], 'entries': []}
        except Exception as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            return {'headers': [], 'entries': []}


class AzureOpenAIAPI(AbstractLLMAPI):
    """Azure OpenAI API wrapper (Azure OpenAI endpoint)."""
    available_models = (
        "gpt-35-turbo",
        "gpt-35-turbo-0301",
        "gpt-35-turbo-16K",
        "gpt-4",
        "gpt-4-32K",
        "gpt-4-turbo-128k",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-11-20-chat",
        "gpt-4.5-preview",
        # 1M context (versioned)
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        # o-series (versioned; require api-version 2024-12-01-preview)
        "o1-2024-12-17",
        "o3-mini-2025-01-31",
        "o4-mini-2025-04-16",
        "o4-mini-2025-04-16-chat",
        "o1",
        "o1-preview",
        "o3",
        "o4-mini",
        "o3-mini",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
    )

    default_judge_model = "gpt-4o-2024-08-06"
    
    def __init__(
            self,
            model: str,
            system_prompt: str
            ) -> None:
        super().__init__(model, system_prompt)
        if model not in self.available_models:
            logger.error(f"Model '{model}' is not in the list of available models: {self.available_models}")
            raise ValueError(f"Model '{model}' is not in the list of available models: {self.available_models}")
        
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.resource_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.max_retries = 2
        self.retry_secs = 3

        if self.resource_endpoint and self.resource_endpoint.endswith('/'):
            self.resource_endpoint = self.resource_endpoint.rstrip('/')

        missing = []
        if not self.api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.api_version:
            missing.append("AZURE_OPENAI_API_VERSION or API_VERSION")
        if not self.resource_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        logger.info(f"AzureOpenAIAPI initialized: model={model}")

        self._cost_guard_acknowledged = os.getenv("RECITE_CONFIRM_PAID_API", "").lower() in ("1", "true", "yes")
        if not self._cost_guard_acknowledged:
            logger.warning(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  COST GUARD: AzureOpenAIAPI initialized without opt-in.       ║\n"
                "║  Set RECITE_CONFIRM_PAID_API=1 to allow paid API calls.     ║\n"
                "║  Without this, all calls will be blocked with an error.     ║\n"
                "╚══════════════════════════════════════════════════════════════╝"
            )

    def __call__(
            self,
            prompt: str,
            system_prompt: str = None,
            temperature: float = 0
            ) -> str:
        if not self._cost_guard_acknowledged:
            raise RuntimeError(
                "COST GUARD: AzureOpenAIAPI call blocked. This API costs real money "
                "(~$30-60 per 3K samples). Set environment variable "
                "RECITE_CONFIRM_PAID_API=1 to acknowledge costs and proceed."
            )
        if system_prompt is None:
            system_prompt = self.system_prompt

        deployment_id = self.model
        url = f"{self.resource_endpoint}/openai/deployments/{deployment_id}/chat/completions?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        body = json.dumps({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        })

        base_delay = float(self.retry_secs)
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Calling Azure OpenAI API at {url}")
                response = requests.post(url, headers=headers, data=body, timeout=60)
                response.raise_for_status()
                resp_json = response.json()
                if "choices" in resp_json and resp_json["choices"]:
                    raw = resp_json["choices"][0]["message"].get("content")
                    output = (raw.strip() if isinstance(raw, str) else (str(raw) if raw is not None else ""))
                    logger.info("Azure OpenAI API call successful.")
                    return output
                else:
                    logger.error(f"Azure OpenAI API response missing 'choices': {resp_json}")
                    return ""
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 400:
                    logger.error(
                        "Azure OpenAI API 400 Bad Request (do not retry). "
                        "Often caused by input exceeding context/token limit. Request body length: %s chars",
                        len(body),
                    )
                    raise
                logger.error(f"Azure OpenAI API call failed: {e}")
                if attempt >= self.max_retries:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Azure OpenAI API call failed: {e}")
                if attempt >= self.max_retries:
                    raise
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                time.sleep(delay)

    def tabulate_eligibility(
            self,
            prompt_template: str,
            *format_args,
            **format_kwargs
            ) -> str:
        prompt = prompt_template.format(*format_args, **format_kwargs)
        llm_response = self(prompt, self.system_prompt)
        return llm_response.strip()

    def tabulate_eligibility_json(
            self,
            prompt_template: str,
            *format_args,
            **format_kwargs
            ) -> dict:
        prompt = prompt_template.format(*format_args, **format_kwargs)
        llm_response = self(prompt, self.system_prompt)

        try:
            result = self.robust_json_parse(llm_response)
            if isinstance(result, dict) and 'headers' in result and 'entries' in result:
                return result
            else:
                logger.error("Response missing 'headers' or 'entries'.")
                return {'headers': [], 'entries': []}
        except Exception as e:
            logger.error(f"Failed to parse response as JSON: {e}")
            return {'headers': [], 'entries': []}

