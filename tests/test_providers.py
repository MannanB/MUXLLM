# python -m unittest discover -s tests -t .

import unittest

from muxllm import LLM, Provider

class TestProvider(unittest.TestCase):
    def test_openai_provider(self):
        llm = LLM(Provider.openai, "gpt-3.5-turbo")
        response = llm.ask("Translate 'Hola, como estas?' to english")
        self.assertEqual("how are you?" in response.message, True)

    def test_groq_provider(self):
        llm = LLM(Provider.groq, "llama3-8b-instruct")
        response = llm.ask("Translate 'Hola, como estas?' to english")
        self.assertEqual("how are you?" in response.message, True)

    def test_fireworks_provider(self):
        llm = LLM(Provider.fireworks, "llama3-8b-instruct")
        response = llm.ask("Translate 'Hola, como estas?' to english")
        self.assertEqual("how are you?" in response.message, True)

    def test_anthropic_provider(self):
        llm = LLM(Provider.anthropic, "claude-3-5-sonnet")
        response = llm.ask("Translate 'Hola, como estas?' to english", max_tokens=500)
        self.assertEqual("how are you?" in response.message, True)

    def test_google_provider(self):
        llm = LLM(Provider.google, "gemini-1.5-pro")
        response = llm.ask("Translate 'Hola, como estas?' to english")
        print(response.message)
        self.assertEqual("how are you?" in response.message, True)

if __name__ == '__main__':
    unittest.main()