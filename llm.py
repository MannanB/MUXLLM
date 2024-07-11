from providers.factory import Provider, create_provider
from prompt import Prompt
from typing import Optional, Union
import json

'''
# usage

llm = LLM(Provider.groq, "llama3-8b-instruct", api_key="your-api-key", system_prompt="You are a helpful translator that translates from spanish to english")
english = llm.ask("Translate "Hola, como estas?" to english")

# example with single prompt

llm = SinglePromptLLM(Provider.groq, "llama3-8b-instruct", "Translate {{spanish}} to english", system_prompt="You are a helpful translator that translates from spanish to english")

english = llm.ask(spanish="Hola, como estas?")

'''

class LLM:
    def __init__(self, provider: Provider, model : str,  api_key : Optional[str] = None, system_prompt : Optional[Union[str, Prompt]] = None):
        self.provider = create_provider(provider, api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.history = []

        if system_prompt is not None:
            self.history.append({"role": "system", "content": str(system_prompt)})

    def __call__(self, messages: list, **kwargs):
        return self.provider.get_response(messages, self.model, **kwargs)

    def save_history(self, fp : str):
        with open(fp, "w") as f:
            json.dump(self.history, f, indent=4)

    def load_history(self, fp : str):
        with open(fp, "r") as f:
            self.history = json.load(f)

    def ask(self, prompt: Union[str, Prompt], system_prompt : Optional[Union[str, Prompt]] = None, **kwargs):
        if isinstance(prompt, Prompt):
            prompt = prompt.get(**kwargs)

        messages = []

        if system_prompt:
            if isinstance(system_prompt, Prompt):
                system_prompt = system_prompt.get(**kwargs)
            messages.append({"role": "system", "content": system_prompt})
        elif self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.provider.get_response(messages, self.model, **kwargs)

        return response.choices[0].message.content
    
    def chat(self, prompt: Union[str, Prompt], **kwargs):
        if isinstance(prompt, Prompt):
            prompt = prompt.get(**kwargs)

        self.history.append({"role": "user", "content": prompt})

        response = self.provider.get_response(self.history, self.model, **kwargs)

        self.history.append({"role": "assistant", "content": response.choices[0].message.content})

        return response.choices[0].message.content

class SinglePromptLLM(LLM):
    def __init__(self, provider: Provider, model : str, prompt : Union[str, Prompt], system_prompt : Optional[Union[str, Prompt]] = None, api_key : Optional[str] = None, **kwargs):
        super().__init__(provider, model, api_key=api_key, system_prompt=system_prompt)
        if isinstance(prompt, Prompt):
            prompt = prompt.get(**kwargs)
        self.prompt = prompt

    def ask(self, **kwargs):
        return super().ask(self.prompt, **kwargs)
