

# MUXLLM

  

muxllm is a python library designed to be an all-in-one wrapper for using LLMs via various cloud providers as well as local inference (WIP). Its main purpose is to be a unified API that allows for hot-swapping between LLMs and between cloud/local inference. It uses a simple interface with built-in chat and prompting capabilities for easy use.
  

Install via pip
```
pip install muxllm
```

[Docs](https://github.com/MannanB/MUXLLM)

# Documentation

  

Basic Usage
----
```python
from muxllm import LLM, Provider

llm = LLM(Provider.openai, "gpt-4")

response = llm.ask("Translate 'Hola, como estas?' to english")

print(response.message) # Hello, how are you?
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
...
llm.load_history("./history.json")
```
Function calling (only for function-calling enabled models)

```python
tools = [...]
llm = LLM(Provider.fireworks, "firefunction-v2")

resp = llm.chat("What is 5*5",
                tools=tools,
                tool_choice={"type": "function"})
                
# muxllm returns a list of ToolCalls
tool_call = resp.tools[0]

# call function with args
tool_response = ...

# the function response is not automatically added to history, so you must manually add it
llm.add_tool_response(tool_call, tool_response)

resp = llm.chat("what did the tool tell you?")
print(resp.message) # 25
```

Each ToolCall contains the name of the tool (```tool_call.name```) and its arguments in a dict (```tool_call.args```). Arguments are always passed as strings, so integers, floats, etc, must be parsed in the function.

In order to use tools you need to supply a dict containing information about each tool that you have available. Check out the [Open AI Guide](https://platform.openai.com/docs/guides/function-calling) to see how to do this.

Backend API usage
----
If you need more fine control, you may choose to directly call the provider's api. There are multiple ways to do this.
1. calling the LLM class
```python
from muxllm import LLM, Provider
llm = LLM(Provider.openai, "gpt-4")
response = llm(messages={...})
print(response.message)
```
2. Using the Provider factory
```python
from muxllm.providers.factory import Provider, create_provider
provider = create_provider(Provider.groq)
response = provider.get_response(messages={...}, model="llama3-8b-instruct")
print(response.message)
```
There may be some edge cases or features of a certain provider that muxllm doesn't cover. In that case, you may want to see the raw response returned directly from either their web API or SDK. This can be done via ```LLMResponse.raw_response``` (note that LLMResponse is what is returned any time you call the LLM, whether it is through .chat, .ask, or any of the above functions)
```python
from muxllm import LLM, Provider
llm = LLM(Provider.openai, "gpt-4")
response = llm(messages={...})
print(response.raw_response)

assert response.message == response.raw_response.choices[0].message
```

Streaming
----
As of right now streaming support is unavailable. However, it is a planned feature.

Async
---
Async is only possible through the provider class as of right now
```python
from muxllm.providers.factory import Provider, create_provider
provider = create_provider(Provider.groq)
# in some async function
response = await provider.get_response_async(messages={...}, model="...")
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

response = llm.ask(myprompt, spanish="Hola, como estas?").content
```
Prompts inside txt files
```python
from muxllm import LLM, Provider, Prompt

llm  =  LLM(Provider.openai, "gpt-3.5-turbo")
# muxllm will look for prompt files in cwd and ./prompts if that folder exists
# You can also provide a direct or relative path to the txt file
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

Tools with muxllm
-- 
Muxllm provides a simple way to automatically create the tools dictionary as well as easily call the functions that the LLM requests.
To use this, you first a create a 	```ToolBox``` that contains all of your tools. Then, using the ```tool``` decorator, you can define functions as tools that you want to use. 
```python
from muxllm.tools import tool, ToolBox, Param

my_tools = ToolBox()

@tool("get_current_weather", my_tools, "Get the current weather", [
    Param("location", "string", "The city and state, e.g. San Francisco, CA"),
    Param("fmt", "string", "The temperature unit to use. Infer this from the users location.")
])
def get_current_weather(location, fmt):
    return f"It is sunny in {location} according to the weather forecast in {fmt}"
```
Note that for the Param class, the second argument is the type of the argument. The possible types are defined here: https://json-schema.org/understanding-json-schema/reference/type

Once you have created each tool, you can then easily convert it to the tools dictionary and pass it to an LLM.
```python
tools_dict = my_tools.to_dict()
response = llm.chat("What is the weather in San Francisco, CA in fahrenheit", tools=tools_dict)
```
Finally, you can use the ```ToolBox``` to invoke the tool and get a response
```python
tool_call = response.tools[0]
tool_resp = my_tools.invoke_tool(tool_call)
llm.add_tool_response(tool_call, tool_resp)
```
Its also possible to have multiple ```ToolBox```s and then combine them. This is useful if you want to remove or add certain tools from the LLM dynamically.
```python
coding_tools = ToolBox()
...
writing_tools = ToolBox()
...
research_tools = ToolBox()
...
# When passing the tools to the LLM
all_tools = coding_tools.to_dict() + writing_tools.to_dict() + research_tools.to_dict()
```

Providers
==
Currently the following providers are available: openai, groq, fireworks, Google Gemini, Anthropic

Local inference /w huggingface and llama.cpp are planned in the future

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

Future Plans
===

* Adding cost tracking / forecasting (I.E. llm.get_cost(...))
* Support for Local Inference
* Seamless async and streaming support
