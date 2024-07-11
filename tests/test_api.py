# python -m unittest discover -s tests -t .

import unittest

from muxllm import LLM, Prompt, Provider
from muxllm.llm import SinglePromptLLM
import os

tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply_numbers",
            "description": "multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]

class TestLLM(unittest.TestCase):
    def test_history(self):
        llm = LLM(Provider.openai, "gpt-3.5-turbo", system_prompt="You are a helpful assistant that answers questions")
        llm.chat("What is the capital of France?")
        llm.save_history("history.json")
        llm.load_history("history.json")
        self.assertEqual(llm.history[1]["content"], "What is the capital of France?")
        self.assertEqual(llm.history[2]["role"], "assistant")
        os.remove("history.json")

    def test_prompts(self):
        prompt = Prompt("My very cool prompt with {{default}} argument, {{changing}} argument, and {{another}} argument", default="im setting this default")
        self.assertEqual(prompt.get(changing="this is changing", another="another"), "My very cool prompt with im setting this default argument, this is changing argument, and another argument")
        self.assertEqual(prompt.get(), "My very cool prompt with im setting this default argument, {{changing}} argument, and {{another}} argument")

    def test_ask(self):
        llm = LLM(Provider.openai, "gpt-3.5-turbo")
        response = llm.ask("Translate {{spanish}} to english", spanish="Hola, como estas?").content
        self.assertEqual("Hello, how are you?" in response, True)

    def test_single_prompt(self):
        llm = SinglePromptLLM(Provider.fireworks, "llama3-8b-instruct", "Translate {{spanish}} to english", system_prompt="You are a helpful translator that translates from spanish to english")
        response = llm.ask(spanish="Hola, como estas?").content
        self.assertEqual("Hello, how are you?" in response, True)

    def test_function_calling(self):
        llm = LLM(Provider.fireworks, "firefunction-v2")

        resp = llm.ask("What is 5*5",
                        tools=tools,
                        tool_choice={"type": "function"})
        
        tool_calls = resp.tool_calls[0].function

        self.assertEqual(tool_calls.name, "multiply_numbers")

