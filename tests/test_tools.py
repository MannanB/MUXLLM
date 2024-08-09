# python -m unittest discover -s tests -t .

import unittest

from muxllm import LLM, Provider


TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]

class TestTools(unittest.TestCase):
    def test_openai_tools(self):
        llm = LLM(Provider.openai, "gpt-4-turbo")
        response = llm.chat("What is the weather in San Francisco, CA", tools=TEST_TOOLS)
        self.assertNotEqual(response.tools, None)
        self.assertEqual(response.tools[0].name, "get_current_weather")

        llm.add_tool_response(response.tools[0], "It is sunny in San Francisco, CA")
        response = llm.chat("Please tell me what the tool said")
        self.assertTrue("sunny" in response.message.lower())

    def test_fireworks_tools(self):
        llm = LLM(Provider.fireworks, "firefunction-v2")
        response = llm.chat("What is the weather in San Francisco, CA", tools=TEST_TOOLS)
        self.assertNotEqual(response.tools, None)
        self.assertEqual(response.tools[0].name, "get_current_weather")

        llm.add_tool_response(response.tools[0], "It is sunny in San Francisco, CA")
        response = llm.chat("Please tell me what the tool said")
        self.assertTrue("sunny" in response.message.lower())

    def test_anthropic_tools(self):
        llm = LLM(Provider.anthropic, "claude-3-5-sonnet")
        response = llm.chat("What is the weather in San Francisco, CA", tools=TEST_TOOLS, max_tokens=500)
        self.assertNotEqual(response.tools, None)
        self.assertEqual(response.tools[0].name, "get_current_weather")

        llm.add_tool_response(response.tools[0], "It is sunny in San Francisco, CA")
        response = llm.chat("Please tell me what the tool said", max_tokens=500)
        self.assertTrue("sunny" in response.message.lower())

    def test_google_tools(self):
        llm = LLM(Provider.google, "gemini-1.5-pro")
        response = llm.chat("What is the weather in San Francisco, CA", tools=TEST_TOOLS)
        self.assertNotEqual(response.tools, None)
        self.assertEqual(response.tools[0].name, "get_current_weather")

        llm.add_tool_response(response.tools[0], "It is sunny in San Francisco, CA")
        response = llm.chat("Please tell me what the tool said")
        self.assertTrue("sunny" in response.message.lower())


if __name__ == '__main__':
    unittest.main()