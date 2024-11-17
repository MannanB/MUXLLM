import openai, os
from typing import Optional
from muxllm.providers.base import CloudProvider

model_alias = {
    "gpt-4-turbo" : "gpt-4-turbo-preview",
    "gpt-4-vision" : "gpt-4-vision-preview",
}

class BaseOpenAIProvider(CloudProvider):
    def __init__(self, model_alias : dict, base_url : str, api_key : Optional[str] = None):
        super().__init__(model_alias)

        self.client = openai.Client(base_url=base_url, api_key=api_key)
        self.async_client = openai.AsyncClient(base_url=base_url, api_key=api_key)

        self.client.chat.completions.create

class OpenAIProvider(BaseOpenAIProvider):
    def __init__(self, api_key : Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(model_alias, base_url="https://api.openai.com/v1", api_key=api_key)
