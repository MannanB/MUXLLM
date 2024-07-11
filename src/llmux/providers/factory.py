from enum import Enum
from llmux.providers import pfireworks, popenai, pgroq

# create an enum for the available providers
class Provider(str, Enum):
    openai = "openai"
    groq = "groq"
    fireworks = "fireworks"

# create a factory method to create the correct provider
def create_provider(provider: Provider, api_key=None):
    if provider == Provider.openai:
        return popenai.OpenAIProvider(api_key)
    elif provider == Provider.groq:
        return pgroq.GroqProvider(api_key)
    elif provider == Provider.fireworks:
        return pfireworks.FireworksProvider(api_key)
    else:
        raise ValueError(f"Provider {provider} is not available")