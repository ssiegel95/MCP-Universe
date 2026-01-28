"""
LiteLLM Integration

This module provides integration with LiteLLM, a unified interface for multiple LLM providers.
LiteLLM supports 100+ LLMs including OpenAI, Anthropic, Cohere, Replicate, and more.
Includes support for fallback models and cost tracking.
"""
# pylint: disable=broad-exception-caught
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Union, Optional, Type, List

from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel

from mcpuniverse.common.config import BaseConfig
from mcpuniverse.common.context import Context
from .base import BaseLLM

load_dotenv()

# Import litellm with error handling
try:
    from litellm import completion, completion_cost
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning("litellm package not found. Install it with: pip install litellm")


@dataclass
class LiteLLMConfig(BaseConfig):
    """
    Configuration for LiteLLM models.

    LiteLLM provides a unified interface to 100+ LLMs. It automatically handles
    provider-specific API calls based on the model name prefix.

    Attributes:
        model_name (str): The primary model to use. Format: "provider/model-name"
            Examples: "gpt-4o", "claude-3-5-sonnet-20241022", "gemini/gemini-pro"
            Default: "gpt-4o"
        fallback_models (List[str]): List of fallback models to try if primary fails.
            Default: empty list
        api_key (str): API key for LiteLLM. Can also use provider-specific keys.
            Default: from LITELLM_API_KEY environment variable
        temperature (float): Controls randomness in output (0.0 to 2.0).
            Default: 1.0
        top_p (float): Controls diversity via nucleus sampling (0.0 to 1.0).
            Default: 1.0
        max_tokens (int): Maximum number of tokens to generate.
            Default: 2048
        timeout (int): Request timeout in seconds.
            Default: 60
        api_base (str): Custom API base URL (optional).
            Default: empty string
        api_version (str): API version for certain providers (e.g., Azure).
            Default: empty string
        track_cost (bool): Whether to track and log API costs.
            Default: True
    """
    model_name: str = "gpt-4o"
    fallback_models: List[str] = field(default_factory=list)
    api_key: str = os.getenv("LITELLM_API_KEY", "")
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout: int = 60
    api_base: str = ""
    api_version: str = ""
    track_cost: bool = True


class LiteLLMModel(BaseLLM):
    """
    LiteLLM language model wrapper with fallback support and cost tracking.

    This class provides a unified interface to interact with 100+ LLM providers
    through LiteLLM, including OpenAI, Anthropic, Cohere, Replicate, and more.

    Features:
    - Automatic fallback to alternative models on failure
    - Cost tracking for API usage
    - Tool calling support for compatible models
    - Retry logic with exponential backoff

    Attributes:
        config_class (Type[LiteLLMConfig]): Configuration class for the model.
        alias (str): Alias for the model, used for identification.
        env_vars (List[str]): Environment variables that may be used.
        total_cost (float): Cumulative cost of API calls (if tracking enabled).

    Example:
        >>> config = {
        ...     "model_name": "gpt-4o",
        ...     "fallback_models": ["gpt-4o-mini", "gpt-3.5-turbo"],
        ...     "temperature": 0.7,
        ...     "track_cost": True
        ... }
        >>> model = LiteLLMModel(config)
        >>> response = model.generate([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(f"Total cost: ${model.total_cost:.4f}")
    """
    config_class = LiteLLMConfig
    alias = "litellm"
    env_vars = ["LITELLM_API_KEY"]

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        """
        Initialize the LiteLLMModel instance.

        Args:
            config (Optional[Union[Dict, str]]): Configuration for the model.
                Can be a dictionary or a JSON string. If None, default
                configuration will be used.

        Raises:
            ImportError: If litellm package is not installed.
        """
        super().__init__()
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm package is required but not installed. "
                "Install it with: pip install litellm"
            )
        self.config = LiteLLMModel.config_class.load(config)
        self.total_cost = 0.0

    def _calculate_cost(self, response) -> float:
        """
        Calculate the cost of an API call using LiteLLM's cost tracking.

        Args:
            response: The response object from LiteLLM completion.

        Returns:
            float: The cost in USD, or 0.0 if calculation fails.
        """
        try:
            cost = completion_cost(completion_response=response)
            return cost if cost is not None else 0.0
        except Exception as e:
            self.logger.warning("Failed to calculate cost: %s", e)
            return 0.0

    def _try_model(
            self,
            model_name: str,
            messages: List[dict[str, str]],
            response_format: Type[PydanticBaseModel] = None,
            **kwargs
    ):
        """
        Attempt to generate content using a specific model.

        Args:
            model_name (str): The model to use.
            messages (List[dict[str, str]]): List of message dictionaries.
            response_format (Type[PydanticBaseModel], optional): Pydantic model
                for structured output.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Any, float]: (response, cost) or (None, 0.0) on failure.
        """
        # Build LiteLLM parameters
        litellm_params = {
            "model": model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "timeout": kwargs.get("timeout", self.config.timeout),
        }

        # Add API key if provided
        if self.config.api_key:
            litellm_params["api_key"] = self.config.api_key

        # Add optional parameters
        if self.config.api_base:
            litellm_params["api_base"] = self.config.api_base
        if self.config.api_version:
            litellm_params["api_version"] = self.config.api_version

        # Handle structured output via response_format
        if response_format is not None:
            litellm_params["response_format"] = {"type": "json_object"}
            # Add schema to system message for better results
            schema_str = response_format.model_json_schema()
            schema_instruction = f"\n\nPlease respond with valid JSON matching this schema: {schema_str}"
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += schema_instruction
            else:
                messages.insert(0, {"role": "system", "content": f"You are a helpful assistant.{schema_instruction}"})

        # Handle tool calling
        if "tools" in kwargs:
            litellm_params["tools"] = kwargs.pop("tools")
        if "tool_choice" in kwargs:
            litellm_params["tool_choice"] = kwargs.pop("tool_choice")

        # Merge any additional kwargs
        litellm_params.update(kwargs)

        try:
            response = completion(**litellm_params)

            # Calculate cost if tracking is enabled
            cost = 0.0
            if self.config.track_cost:
                cost = self._calculate_cost(response)
                self.total_cost += cost
                self.logger.info(
                    "Model: %s | Cost: $%.6f | Total: $%.6f",
                    model_name, cost, self.total_cost
                )

            # Handle tool calling responses
            if "tools" in litellm_params:
                return response, cost

            # Extract content from response
            content = response.choices[0].message.content

            # Parse structured output if response_format was provided
            if response_format is not None:
                try:
                    import json
                    parsed_json = json.loads(content)
                    return response_format.model_validate(parsed_json), cost
                except Exception as e:
                    self.logger.error("Failed to parse structured output: %s", e)
                    self.logger.error("Raw content: %s", content)
                    return None, cost

            return content, cost

        except Exception as e:
            self.logger.warning("Model %s failed: %s", model_name, e)
            return None, 0.0

    def _generate(
            self,
            messages: List[dict[str, str]],
            response_format: Type[PydanticBaseModel] = None,
            **kwargs
    ):
        """
        Generates content using LiteLLM with fallback support.

        This method tries the primary model first, then falls back to alternative
        models if specified in the configuration.

        Args:
            messages (List[dict[str, str]]): List of message dictionaries,
                each containing 'role' and 'content' keys.
            response_format (Type[PydanticBaseModel], optional): Pydantic model
                defining the structure of the desired output. If provided,
                uses JSON mode for structured output.
            **kwargs: Additional keyword arguments including:
                - max_retries (int): Maximum number of retry attempts per model (default: 3)
                - base_delay (float): Base delay in seconds for exponential backoff (default: 2.0)
                - tools (list): Tool definitions for function calling
                - tool_choice (str): Tool choice strategy

        Returns:
            Union[str, PydanticBaseModel, dict, None]: Generated content as a string
                if no response_format is provided, a Pydantic model instance if
                response_format is provided, or the full response object if tools
                are used. Returns None if all models fail.
        """
        max_retries = kwargs.pop("max_retries", 3)
        base_delay = kwargs.pop("base_delay", 2.0)

        # Build list of models to try: primary + fallbacks
        models_to_try = [self.config.model_name] + self.config.fallback_models

        # Try each model
        for model_idx, model_name in enumerate(models_to_try):
            self.logger.info(
                "Trying model %d/%d: %s",
                model_idx + 1, len(models_to_try), model_name
            )

            # Retry logic with exponential backoff for each model
            for attempt in range(max_retries + 1):
                result, cost = self._try_model(
                    model_name, messages, response_format, **kwargs
                )

                if result is not None:
                    return result

                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = base_delay * (2 ** attempt)
                    self.logger.info(
                        "Attempt %d/%d failed. Retrying in %.1f seconds...",
                        attempt + 1, max_retries + 1, delay
                    )
                    time.sleep(delay)

            # If we get here, all retries for this model failed
            self.logger.warning(
                "All %d attempts failed for model: %s",
                max_retries + 1, model_name
            )

        # All models failed
        self.logger.error(
            "All models failed: %s",
            ", ".join(models_to_try)
        )
        return None

    def set_context(self, context: Context):
        """
        Set context, including environment variables (API keys).

        Args:
            context (Context): Context object containing environment variables
                and metadata.
        """
        super().set_context(context)
        self.config.api_key = context.get_env("LITELLM_API_KEY", self.config.api_key)

    def support_tool_call(self) -> bool:
        """
        Return a flag indicating if the model supports function/tool calling.

        LiteLLM supports tool calling for compatible models (OpenAI, Anthropic, etc.).

        Returns:
            bool: True, indicating tool calling support.
        """
        return True

    def get_total_cost(self) -> float:
        """
        Get the total cost of all API calls made by this model instance.

        Returns:
            float: Total cost in USD.
        """
        return self.total_cost

    def reset_cost(self):
        """
        Reset the cost tracker to zero.
        """
        self.total_cost = 0.0
        self.logger.info("Cost tracker reset to $0.00")

# Made with Bob
