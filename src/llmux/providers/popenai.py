import openai, os
from typing import Optional
from llmux.providers.base import CloudProvider

model_alias = {}
available_models = [
                "gpt-4-0125-preview",
                "gpt-4-turbo-preview",
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-4-32k-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-1106",
                "gpt-3.5-turbo-0125",
                "gpt-3.5-turbo-16k-0613",
            ]

class BaseOpenAIProvider(CloudProvider):
    def __init__(self, available_models : list, model_alias : dict, base_url : str, api_key : Optional[str] = None):
        super().__init__(available_models, model_alias)

        self.client = openai.Client(base_url=base_url, api_key=api_key)
        self.async_client = openai.AsyncClient(base_url=base_url, api_key=api_key)

class OpenAIProvider(BaseOpenAIProvider):
    def __init__(self, api_key : Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        super().__init__(available_models, model_alias, base_url="https://api.openai.com/v1", api_key=api_key)
