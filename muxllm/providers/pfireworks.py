from muxllm.providers.popenai import BaseOpenAIProvider
from typing import Optional
import os

model_alias = {
    "firefunction-v2": "accounts/fireworks/models/firefunction-v2",
    "mixtral-8x7b-instruct": "accounts/fireworks/models/mixtral-8x7b-instruct",
    "mixtral-8x22b-instruct": "accounts/fireworks/models/mixtral-8x22b-instruct",
    "llama3-8b-instruct": "accounts/fireworks/models/llama-v3-8b-instruct",
    "llama3-70b-instruct": "accounts/fireworks/models/llama-v3-70b-instruct",
    "gemma-7b-instruct": "accounts/fireworks/models/gemma-7b-it",
    "gemma2-9b-instruct": "accounts/fireworks/models/gemma-9b-it",
}

available_models = [] # empty means that all models are available, mostly because there are far too many models on fireworks

class FireworksProvider(BaseOpenAIProvider):
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("FIREWORKS_API_KEY")
        super().__init__(available_models, model_alias, base_url="https://api.fireworks.ai/inference/v1", api_key=api_key)