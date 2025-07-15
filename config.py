
import os
import time
from dotenv import load_dotenv

# Provider-specific libraries & exceptions
import google.generativeai as genai
from openai import OpenAI, RateLimitError as OpenAIRateLimitError
from anthropic import Anthropic, RateLimitError as AnthropicRateLimitError
from google.api_core import exceptions as GoogleAPIErrors

# Load environment variables from .env file
load_dotenv()

class LLMConfig:
    # ... (this class remains unchanged) ...
    """
    A factory class to get a specific, ready-to-use LLM client.
    """
    def __init__(self):
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY")
        }

    def get_client(self, llm_name: str, model: str):
        """
        Returns an initialized, provider-specific client instance.
        """
        llm_name = llm_name.lower()
        api_key = self.api_keys.get(llm_name)

        if not api_key:
            raise ValueError(f"API key for '{llm_name}' not found in .env file.")

        if llm_name == 'openai':
            return OpenAIClient(api_key, model)
        elif llm_name == 'google':
            return GoogleClient(api_key, model)
        elif llm_name == 'anthropic':
            return AnthropicClient(api_key, model)
        else:
            raise ValueError(f"LLM provider '{llm_name}' is not supported.")

# --- Client Classes with **kwargs for flexibility ---

class BaseLLMClient:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, max_retries=3, **kwargs):
        raise NotImplementedError

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                # Pass kwargs directly to the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs 
                )
                return response.choices[0].message.content.strip()
            except OpenAIRateLimitError as e:
                # ... retry logic ...
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else: return "API_RETRY_ERROR"
            except Exception as e:
                return f"UNEXPECTED_ERROR: {e}"
        return "API_RETRY_ERROR"

class GoogleClient(BaseLLMClient):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        genai.configure(api_key=self.api_key)
        safety_settings = [
            {"category": c, "threshold": "BLOCK_NONE"} for c in 
            ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
             "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]
        self.client = genai.GenerativeModel(self.model, safety_settings=safety_settings)

    def generate(self, prompt: str, max_retries=3, **kwargs):
        # Map common kwargs to Google's GenerationConfig
        config_params = {
            "temperature": kwargs.get("temperature"),
            "max_output_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k")
        }
        # Filter out None values
        generation_config = {k: v for k, v in config_params.items() if v is not None}
        
        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(
                    prompt, 
                    generation_config=genai.GenerationConfig(**generation_config)
                )
                return response.text.strip()
            except (GoogleAPIErrors.ServiceUnavailable, GoogleAPIErrors.RetryError) as e:
                # ... retry logic ...
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else: return "API_RETRY_ERROR"
            except Exception as e:
                if 'response was blocked' in str(e): return "BLOCKED_BY_SAFETY_FILTER"
                return f"UNEXPECTED_ERROR: {e}"
        return "API_RETRY_ERROR"

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key, model):
        super().__init__(api_key, model)
        self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, max_retries=3, **kwargs):
        # Anthropic requires max_tokens, so provide a default if not present
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 256 # A sensible default

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return response.content[0].text.strip()
            except AnthropicRateLimitError as e:
                # ... retry logic ...
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else: return "API_RETRY_ERROR"
            except Exception as e:
                return f"UNEXPECTED_ERROR: {e}"
        return "API_RETRY_ERROR"
