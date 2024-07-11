
# MUXLLM

  

muxllm is a python library designed to be an all-in-one wrapper for various cloud providers as well as local inference (WIP)

  
  

[Docs](https://github.com/MannanB/MUXLLM)

# Documentation

  

Basic Usage
----
```python
from muxllm import LLM, Provider

llm = LLM(Provider.openai, "gpt-4")

response = llm.ask("Translate 'Hola, como estas?' to english")

print(response.content) # Hello, how are you?
```

API keys can be passed via the api_key parameter in the LLM class. Otherwise, it will try to get the key from environment variables with the following pattern: [PROVIDER_NAME]_API_KEY

Adding a system prompt
```python
llm = LLM(Provider.openai, "gpt-4", system_prompt="...")
```
Anything passed to kwargs will be passed to the chat completion function for whatever provider you chose
```python
llm.ask("...", temperature=1, top_p=.5)
```
Built-in chat functionality with conversation history
```python
response = llm.chat("how are you?")
print(response.content)
# the previous response has automatically been stored
response = llm.chat("what are you doing right now?") 
llm.save_history("./history.json") # save history if you want to continue conversation later
```
Function calling (only for function-calling enabled models)

```python
tools = [...]
llm = LLM(Provider.fireworks, "firefunction-v2")

resp = llm.chat("What is 5*5",
                tools=tools,
                tool_choice={"type": "function"})

tool_call = resp.tool_calls[0].function
# call function
tool_response = ...
# the function response is not automatically added to history, so you must manually add it if you want the llm to have the tool response in context
llm.history.append({"role": "tool", "content": tool_response})
```

Backend API usage
----
If you need more fine control, you may choose to directly call the provider's api. There are multiple ways to do this.
1. calling the LLM class
```python
from muxllm import LLM, Provider
llm = LLM(Provider.openai, "gpt-4")
response = llm(messages={...})
print(response.choices[0].message.content)
```
2. Using the Provider factory
```python
from muxllm.providers.factory import Provider, create_provider
provider = create_provider(Provider.groq)
response = provider.get_response(messages={...}, model="llama3-8b-instruct")
print(response.choices[0].message.content)
```
Streaming
----
Streaming via LLM class (does not work with .ask or .chat methods as of right now)
```python
from muxllm import LLM, Provider
llm = LLM(Provider.openai, "gpt-4")
response = llm(messages={...}, model="llama3-8b-instruct", stream=True)
for chunk in response:
    print(chunk.choices[0].delta.content)
```
Streaming via provider class
```python
from muxllm.providers.factory import Provider, create_provider
provider = create_provider(Provider.groq)
response = provider.get_response(messages={...}, stream=True)
for chunk in response:
    print(chunk.choices[0].delta.content)
```
Async
---
Async is only possible through the provider class as of right now
```python
from muxllm.providers.factory import Provider, create_provider
provider = create_provider(Provider.groq)
# in some async function
response = await provider.get_response_async(messages={...})
```
Prompting with muxllm
--
muxllm provides a simple way to add pythonic prompting
For any plain text file or string surrounding a variable with {{ ... }} will allow you to reference the variable when using the LLM class

Example Usage
```python
from muxllm import LLM, Provider

llm  =  LLM(Provider.openai, "gpt-3.5-turbo")
myprompt = "Translate {{spanish}} to english"

response  =  llm.ask(myprompt, spanish="Hola, como estas?").content
```
Prompts inside txt files
```python
from muxllm import LLM, Provider, Prompt

llm  =  LLM(Provider.openai, "gpt-3.5-turbo")
# muxllm will look for prompt files in cwd and ./prompts if that folder exists
response  =  llm.ask(Prompt("translate_prompt.txt"), spanish="Hola, como estas?").content
```

Single Prompt LLMs; A lot of times a single LLM class is only used for one prompt. SinglePromptLLM is a subclass of LLM but can only use one prompt given in the constructor
```python
from muxllm.llm import SinglePromptLLM
from muxllm import Prompt, Provider

llm = SinglePromptLLM(Provider.openai, "gpt-3.5-turbo", prompt="translate {{spanish}} to english")

# via file
llm = SinglePromptLLM(Provider.openai, "gpt-3.5-turbo", prompt=Prompt("translate_prompt.txt"))

print(llm.ask(spanish="hola, como estas?").content)
```

Providers
==
Currently the following providers are available: openai, groq, fireworks
Google Gemini, Anthropic, and local inference /w huggingface and llama.cpp are planned in the future

Model Alias
---
Fireworks, Groq, and local inference have common models. For the sake of generalization, these have been given aliases that you may choose to use if you don't want to use the specific model name for that provider. This gives the benefit of being interchangeable between providers without having to change the model name
| Name                   | Fireworks                                        | Groq               | HuggingFace |
| ---------------------- | ------------------------------------------------ | ------------------ | ----------- |
| llama3-8b-instruct     | accounts/fireworks/models/llama-v3-8b-instruct   | llama3-8b-8192     | WIP         |
| llama3-70b-instruct    | accounts/fireworks/models/llama-v3-70b-instruct  | llama3-70b-8192    | WIP         |
| mixtral-8x7b-instruct  | accounts/fireworks/models/mixtral-8x7b-instruct  | mixtral-8x7b-32768 | WIP         |
| gemma-7b-instruct      | accounts/fireworks/models/gemma-7b-it            | gemma-7b-it        | WIP         |
| gemma2-9b-instruct     | accounts/fireworks/models/gemma-9b-it            | gemma-9b-it        | WIP         |
| firefunction-v2        | accounts/fireworks/models/firefunction-v2        | N/A                | WIP         |
| mixtral-8x22b-instruct | accounts/fireworks/models/mixtral-8x22b-instruct | N/A                | WIP         |

```python
# the following are all equivalent, in terms of what model they use
LLM(Provider.fireworks, "llama3-8b-instruct")
LLM(Provider.groq, "llama3-8b-instruct")
LLM(Provider.fireworks, "accounts/fireworks/models/llama-v3-8b-instruct")
LLM(Provider.groq, "llama3-8b-8192")
```


