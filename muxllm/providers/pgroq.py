from groq import Groq, AsyncGroq
from typing import Optional
import os

from muxllm.providers.base import CloudProvider

model_alias = {
    "llama3-8b-instruct": "llama3-8b-8192",
    "llama3-70b-instruct": "llama3-70b-8192",
    "mixtral-8x7b-instruct": "mixtral-8x7b-32768",
    "gemma-7b-instruct": "gemma-7b-it",
    "gemma2-9b-instruct": "gemma-9b-it",
}

class GroqProvider(CloudProvider):
    def __init__(self, api_key : Optional[str] = None):
        super().__init__(model_alias)
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.async_client = AsyncGroq(api_key=api_key)