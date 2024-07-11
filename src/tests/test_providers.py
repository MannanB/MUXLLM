# python -m unittest discover -s tests -t .

import unittest

from llmux import LLM, Provider

class TestProvider(unittest.TestCase):
    def test_openai_provider(self):
        llm = LLM(Provider.openai, "gpt-3.5-turbo")
        response = llm.ask("Translate 'Hola, como estas?' to english")
        self.assertEqual("Hello, how are you?" in response.content, True)

    def test_groq_provider(self):
        llm = LLM(Provider.groq, "llama3-8b-instruct")
        response = llm.ask("Translate 'Hola, como estas?' to english")
        self.assertEqual("Hello, how are you?" in response.content, True)

    def test_fireworks_provider(self):
        llm = LLM(Provider.fireworks, "llama3-8b-instruct")
        response = llm.ask("Translate 'Hola, como estas?' to english")
        self.assertEqual("Hello, how are you?" in response.content, True)

if __name__ == '__main__':
    unittest.main()