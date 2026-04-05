"""
llmapis.py

This script defines a lightweight wrapper around the OpenAI Chat Completions API
using the official `openai` Python client. It supports interaction with GPT-4.1 family
models, with configurable model selection and API key loading via environment variables.

Abbasi Lab, UCSF
"""


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
    """Abstract class for instantiating objects that serve as
    callable LLM APIs.
    """
    def __init__(
            self,
            model: str,
            system_prompt: str,
            *args,
            **kwargs
            ) -> None:
        """
        """
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
        """Cleans and parses LLM JSON output, handling common formatting issues.
        """
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
        """Given a prompt template and format arguments, call the LLM and return the response.
        
        Args:
            prompt_template (str): Template string with placeholders
            *format_args: Positional arguments for template formatting
            **format_kwargs: Keyword arguments for template formatting
            
        Returns:
            str: Raw LLM response
        """
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
        """Given a prompt template and format arguments, call the LLM and parse JSON response.
        
        Args:
            prompt_template (str): Template string with placeholders
            *format_args: Positional arguments for template formatting
            **format_kwargs: Keyword arguments for template formatting
            
        Returns:
            dict: Parsed JSON response with 'headers' and 'entries' keys
        """
        prompt = prompt_template.format(*format_args, **format_kwargs)
        logger.debug("Calling LLM for tabulate_eligibility_json.")
        llm_response = self(prompt, self.system_prompt)

        try:
            result = self.robust_json_parse(llm_response)
            if isinstance(result, dict) and 'headers' in result and 'entries' in result:
                logger.info("LLM returned valid tabulated eligibility JSON.")
                return result
            else:
                logger.error("LLM response missing 'headers' or 'entries'. Returning empty result.")
                return {'headers': [], 'entries': []}
        except Exception as e:
            logger.error(f"Failed to robustly parse LLM response as JSON: {e}")
            return {'headers': [], 'entries': []}


class OpenAIAPI(AbstractLLMAPI):
    """A simple wrapper around the OpenAI Chat Completions API.
    
    This class initializes an OpenAI client and allows calling the API
    with a prompt via the `__call__` method.
    """
    def __init__(
            self,
            model: str,
            system_prompt: str
            ) -> None:
        """Initialize the OpenAI API wrapper.

        Args:
            model (str): The default model to use for completions.
            system_prompt (str): The system message to guide the model's behavior.
        """
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
        """Call the OpenAI Chat Completion API with a user prompt.

        Args:
            prompt (str): The user's prompt.
            system_prompt (str): The system message to guide the model's behavior.
            temperature (float): Sampling temperature.

        Returns:
            str: The model's reply.
        """
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
        """Implementation for OpenAIAPI. Calls the LLM and returns the raw response.
        
        Args:
            prompt_template (str): Template string with placeholders
            *format_args: Positional arguments for template formatting
            **format_kwargs: Keyword arguments for template formatting
            
        Returns:
            str: Raw LLM response
        """
        prompt = prompt_template.format(*format_args, **format_kwargs)
        logger.debug("Calling OpenAIAPI.tabulate_eligibility.")
        llm_response = self(prompt, self.system_prompt)
        return llm_response.strip()

    def tabulate_eligibility_json(
            self,
            prompt_template: str,
            *format_args,
            **format_kwargs
            ) -> dict:
        """Implementation for OpenAIAPI. Calls the LLM and parses JSON response.
        
        Args:
            prompt_template (str): Template string with placeholders
            *format_args: Positional arguments for template formatting
            **format_kwargs: Keyword arguments for template formatting
            
        Returns:
            dict: Parsed JSON response
        """
        prompt = prompt_template.format(*format_args, **format_kwargs)
        logger.debug("Calling OpenAIAPI.tabulate_eligibility_json.")
        llm_response = self(prompt, self.system_prompt)
        
        try:
            result = self.robust_json_parse(llm_response)
            if isinstance(result, dict) and 'headers' in result and 'entries' in result:
                logger.info("OpenAIAPI returned valid tabulated eligibility JSON.")
                return result
            else:
                logger.error("OpenAIAPI response missing 'headers' or 'entries'. Returning empty result.")
                return {'headers': [], 'entries': []}
        except Exception as e:
            logger.error(f"Failed to robustly parse OpenAIAPI response as JSON: {e}")
            return {'headers': [], 'entries': []}


class UCSFVersaAPI(AbstractLLMAPI):
    """Wrapper for the UCSF Versa Azure OpenAI API.
    Expects the following environment variables:
      - UCSF_API_KEY
      - UCSF_API_VER
      - UCSF_RESOURCE_ENDPOINT
    """
    # Available Azure OpenAI models (class attribute for importability)
    # Versioned names from Usage_data that ping OK: gpt-4.1-2025-04-14, gpt-4.1-mini-2025-04-14, gpt-4o-*
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
        # Claude (from Usage_data; may 404 on some endpoints)
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
    )
    
    # Default judge model (recommended for evaluation tasks)
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
        
        self.api_key = os.getenv("UCSF_API_KEY")
        self.api_version = os.getenv("UCSF_API_VER")
        self.resource_endpoint = os.getenv("UCSF_RESOURCE_ENDPOINT")
        self.max_retries = 2
        self.retry_secs = 3

        # Remove trailing slash if present
        if self.resource_endpoint and self.resource_endpoint.endswith('/'):
            self.resource_endpoint = self.resource_endpoint.rstrip('/')

        # Validate env vars
        missing = []
        if not self.api_key:
            missing.append("UCSF_API_KEY")
        if not self.api_version:
            missing.append("UCSF_API_VER or API_VERSION")
        if not self.resource_endpoint:
            missing.append("UCSF_RESOURCE_ENDPOINT")
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        logger.info(f"UCSFVersaAPI client initialized for deployment '{model}' at '{self.resource_endpoint}' with API version '{self.api_version}'.")

        # Cost guard: require explicit opt-in for paid API usage
        self._cost_guard_acknowledged = os.getenv("RECITE_CONFIRM_PAID_API", "").lower() in ("1", "true", "yes")
        if not self._cost_guard_acknowledged:
            logger.warning(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  COST GUARD: UCSFVersaAPI initialized without opt-in.       ║\n"
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
                "COST GUARD: UCSFVersaAPI call blocked. This API costs real money "
                "(~$30-60 per 3K samples). Set environment variable "
                "RECITE_CONFIRM_PAID_API=1 to acknowledge costs and proceed."
            )
        if system_prompt is None:
            system_prompt = self.system_prompt

        deployment_id = self.model  # In Azure, this is the deployment name
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

        base_delay = float(self.retry_secs)  # First delay in seconds
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Calling UCSF Versa API at {url}")
                response = requests.post(url, headers=headers, data=body, timeout=60)
                response.raise_for_status()
                resp_json = response.json()
                if "choices" in resp_json and resp_json["choices"]:
                    raw = resp_json["choices"][0]["message"].get("content")
                    # Ensure we always return str (avoids AssertionError from downstream assert isinstance(..., str))
                    output = (raw.strip() if isinstance(raw, str) else (str(raw) if raw is not None else ""))
                    logger.info("UCSF Versa API call successful.")
                    return output
                else:
                    logger.error(f"UCSF Versa API response missing 'choices': {resp_json}")
                    return ""
            except requests.exceptions.HTTPError as e:
                # 400 Bad Request: don't retry (invalid request, e.g. input too long)
                if e.response is not None and e.response.status_code == 400:
                    logger.error(
                        "UCSF Versa API 400 Bad Request (do not retry). "
                        "Often caused by input exceeding context/token limit. Request body length: %s chars",
                        len(body),
                    )
                    raise
                logger.error(f"UCSF Versa API call failed: {e}")
                if attempt >= self.max_retries:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                time.sleep(delay)
            except Exception as e:
                logger.error(f"UCSF Versa API call failed: {e}")
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
        """Implementation for UCSFVersaAPI. Calls the LLM and returns the raw response.
        
        Args:
            prompt_template (str): Template string with placeholders
            *format_args: Positional arguments for template formatting
            **format_kwargs: Keyword arguments for template formatting
            
        Returns:
            str: Raw LLM response
        """
        prompt = prompt_template.format(*format_args, **format_kwargs)
        logger.debug("Calling UCSFVersaAPI.tabulate_eligibility.")
        llm_response = self(prompt, self.system_prompt)
        return llm_response.strip()

    def tabulate_eligibility_json(
            self,
            prompt_template: str,
            *format_args,
            **format_kwargs
            ) -> dict:
        """Implementation for UCSFVersaAPI. Calls the LLM and parses JSON response.
        
        Args:
            prompt_template (str): Template string with placeholders
            *format_args: Positional arguments for template formatting
            **format_kwargs: Keyword arguments for template formatting
            
        Returns:
            dict: Parsed JSON response
        """
        prompt = prompt_template.format(*format_args, **format_kwargs)
        logger.debug("Calling UCSFVersaAPI.tabulate_eligibility_json.")
        llm_response = self(prompt, self.system_prompt)

        try:
            result = self.robust_json_parse(llm_response)
            if isinstance(result, dict) and 'headers' in result and 'entries' in result:
                logger.info("UCSFVersaAPI returned valid tabulated eligibility JSON.")
                return result
            else:
                logger.error("UCSFVersaAPI response missing 'headers' or 'entries'. Returning empty result.")
                return {'headers': [], 'entries': []}
        except Exception as e:
            logger.error(f"Failed to robustly parse UCSFVersaAPI response as JSON: {e}")
            return {'headers': [], 'entries': []}

