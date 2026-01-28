import unittest
import pytest
from unittest.mock import patch
from pydantic import BaseModel
from dotenv import load_dotenv
from mcpuniverse.llm.litellm import LiteLLMModel, LiteLLMConfig
from mcpuniverse.common.context import Context

# Load environment variables from .env file
load_dotenv()


class CodeResponse(BaseModel):
    """Pydantic model for structured output testing."""
    code: str
    explanation: str


class TestLiteLLM(unittest.TestCase):
    """Test suite for LiteLLMModel."""

    def test_initialization_default_config(self):
        """Test model initialization with default configuration."""
        model = LiteLLMModel()
        self.assertIsNotNone(model.config)
        self.assertEqual(model.config.model_name, "Azure/gpt-5-mini-2025-08-07")
        self.assertEqual(model.config.temperature, 1.0)
        self.assertEqual(model.config.max_tokens, 2048)
        self.assertTrue(model.config.track_cost)
        self.assertEqual(model.total_cost, 0.0)

    def test_initialization_custom_config(self):
        """Test model initialization with custom configuration."""
        config = {
            "model_name": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 1024,
            "track_cost": False,
            "fallback_models": ["gpt-4o-mini", "gpt-3.5-turbo"]
        }
        model = LiteLLMModel(config)
        self.assertEqual(model.config.model_name, "gpt-4o")
        self.assertEqual(model.config.temperature, 0.7)
        self.assertEqual(model.config.max_tokens, 1024)
        self.assertFalse(model.config.track_cost)
        self.assertEqual(len(model.config.fallback_models), 2)

    def test_list_undefined_env_vars(self):
        """Test listing undefined environment variables."""
        # Test with empty API key
        with patch.dict('os.environ', {'LITELLM_API_KEY': ''}):
            model = LiteLLMModel()
            undefined_vars = model.list_undefined_env_vars()
            self.assertIn("LITELLM_API_KEY", undefined_vars)

        # Test with valid API key
        with patch.dict('os.environ', {'LITELLM_API_KEY': 'test_key_123'}):
            model = LiteLLMModel()
            undefined_vars = model.list_undefined_env_vars()
            self.assertNotIn("LITELLM_API_KEY", undefined_vars)

    def test_set_context(self):
        """Test setting context with environment variables."""
        context = Context(env={"LITELLM_API_KEY": "context_api_key"})
        model = LiteLLMModel()
        model.set_context(context)
        self.assertEqual(model.config.api_key, "context_api_key")

    def test_support_tool_call(self):
        """Test that LiteLLM supports tool calling."""
        model = LiteLLMModel()
        self.assertTrue(model.support_tool_call())

    def test_cost_tracking_methods(self):
        """Test cost tracking functionality."""
        model = LiteLLMModel()
        
        # Initial cost should be 0
        self.assertEqual(model.get_total_cost(), 0.0)
        
        # Manually set cost for testing
        model.total_cost = 0.05
        self.assertEqual(model.get_total_cost(), 0.05)
        
        # Reset cost
        model.reset_cost()
        self.assertEqual(model.get_total_cost(), 0.0)

    @pytest.mark.skip(reason="Requires valid API key and makes real API calls")
    def test_basic_generation(self):
        """Test basic text generation with LiteLLM."""
        config = {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 100,
            "track_cost": True
        }
        model = LiteLLMModel(config)
        
        system_message = "You are a helpful assistant."
        user_message = "Say 'Hello, World!' and nothing else."
        
        response = model.get_response(system_message, user_message)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(f"Response: {response}")
        print(f"Total cost: ${model.get_total_cost():.6f}")

    @pytest.mark.skip(reason="Requires valid API key and makes real API calls")
    def test_structured_output(self):
        """Test structured output generation with Pydantic model."""
        config = {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 500,
            "track_cost": True
        }
        model = LiteLLMModel(config)
        
        system_message = "You are a professional Python developer."
        user_message = """
        Write a simple Python function to calculate the factorial of a number.
        Provide the code and a brief explanation in JSON format.
        """
        
        response = model.get_response(
            system_message, 
            user_message, 
            response_format=CodeResponse
        )
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, CodeResponse)
        self.assertIsInstance(response.code, str)
        self.assertIsInstance(response.explanation, str)
        self.assertGreater(len(response.code), 0)
        self.assertGreater(len(response.explanation), 0)
        
        print(f"Code:\n{response.code}")
        print(f"\nExplanation:\n{response.explanation}")
        print(f"\nTotal cost: ${model.get_total_cost():.6f}")

    @pytest.mark.skip(reason="Requires valid API key and makes real API calls")
    def test_fallback_models(self):
        """Test fallback model functionality."""
        config = {
            "model_name": "invalid-model-that-does-not-exist",
            "fallback_models": ["gpt-4o-mini", "gpt-3.5-turbo"],
            "temperature": 0.7,
            "max_tokens": 100,
            "track_cost": True
        }
        model = LiteLLMModel(config)
        
        system_message = "You are a helpful assistant."
        user_message = "Say 'Fallback successful!' and nothing else."
        
        # Should fall back to gpt-4o-mini after invalid model fails
        response = model.get_response(system_message, user_message)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(f"Response: {response}")
        print(f"Total cost: ${model.get_total_cost():.6f}")

    @pytest.mark.skip(reason="Requires valid API key and makes real API calls")
    def test_generate_with_messages(self):
        """Test generation using message list format."""
        config = {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 150,
            "track_cost": True
        }
        model = LiteLLMModel(config)
        
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 15 + 27?"},
            {"role": "assistant", "content": "15 + 27 = 42"},
            {"role": "user", "content": "Now multiply that by 2."}
        ]
        
        response = model.generate(messages)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(f"Response: {response}")
        print(f"Total cost: ${model.get_total_cost():.6f}")

    @pytest.mark.skip(reason="Requires valid API key and makes real API calls")
    def test_cost_tracking_accumulation(self):
        """Test that costs accumulate across multiple API calls."""
        config = {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 50,
            "track_cost": True
        }
        model = LiteLLMModel(config)
        
        # Make first call
        response1 = model.get_response("You are helpful.", "Say hello.")
        cost_after_first = model.get_total_cost()
        self.assertGreater(cost_after_first, 0.0)
        
        # Make second call
        response2 = model.get_response("You are helpful.", "Say goodbye.")
        cost_after_second = model.get_total_cost()
        
        # Cost should have increased
        self.assertGreater(cost_after_second, cost_after_first)
        
        print(f"Cost after first call: ${cost_after_first:.6f}")
        print(f"Cost after second call: ${cost_after_second:.6f}")
        print(f"Total accumulated cost: ${cost_after_second:.6f}")

    @pytest.mark.skip(reason="Requires valid API key and makes real API calls")
    def test_different_providers(self):
        """Test using different LLM providers through LiteLLM."""
        # Test with OpenAI
        openai_config = {
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 50
        }
        openai_model = LiteLLMModel(openai_config)
        openai_response = openai_model.get_response(
            "You are helpful.", 
            "Say 'OpenAI works!'"
        )
        self.assertIsNotNone(openai_response)
        print(f"OpenAI Response: {openai_response}")
        
        # Note: To test other providers, you would need their API keys
        # Example for Anthropic Claude:
        # claude_config = {
        #     "model_name": "claude-3-5-sonnet-20241022",
        #     "temperature": 0.7,
        #     "max_tokens": 50
        # }
        # claude_model = LiteLLMModel(claude_config)
        # claude_response = claude_model.get_response(
        #     "You are helpful.", 
        #     "Say 'Claude works!'"
        # )

    def test_config_export(self):
        """Test configuration export functionality."""
        config = {
            "model_name": "gpt-4o",
            "temperature": 0.8,
            "max_tokens": 2000,
            "fallback_models": ["gpt-4o-mini"]
        }
        model = LiteLLMModel(config)
        
        exported_config = model.export_config()
        self.assertIsNotNone(exported_config)
        self.assertIn("name", exported_config)
        self.assertIn("config", exported_config)
        self.assertEqual(exported_config["name"], "LiteLLMModel")
        self.assertIn("model_name", exported_config["config"])
        self.assertEqual(exported_config["config"]["model_name"], "gpt-4o")
        self.assertEqual(exported_config["config"]["temperature"], 0.8)


if __name__ == "__main__":
    unittest.main()

# Made with Bob
