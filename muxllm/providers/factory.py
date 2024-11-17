from enum import Enum
from muxllm.providers import pfireworks, popenai, pgroq, panthropic, pgoogle
import importlib # for local provider
from muxllm.providers.base import CloudProvider

# create an enum for the available providers
class Provider(str, Enum):
    openai = "openai"
    groq = "groq"
    fireworks = "fireworks"
    anthropic = "anthropic"
    google = "google"
    local = "local"

# create a factory method to create the correct provider
def create_provider(provider: Provider, api_key=None) -> CloudProvider:
    if provider == Provider.openai:
        return popenai.OpenAIProvider(api_key)
    elif provider == Provider.groq:
        return pgroq.GroqProvider(api_key)
    elif provider == Provider.fireworks:
        return pfireworks.FireworksProvider(api_key)
    elif provider == Provider.anthropic:
        return panthropic.AnthropicProvider(api_key)
    elif provider == Provider.google:
        return pgoogle.GoogleProvider(api_key)
    elif provider == Provider.local:
        try:
            import llama_cpp
        except:
            raise ValueError("Local provider requires the llama_cpp package to be installed")
        return importlib.import_module("muxllm.providers.plocal").LocalProvider()
    else:
        raise ValueError(f"Provider {provider} is not available")