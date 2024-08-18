# python -m unittest discover -s tests -t .

import unittest

from muxllm import LLM, Provider
from muxllm.tools import tool, Param, ToolBox
from muxllm.providers.base import ToolCall

my_tools = ToolBox()

@tool("get_current_weather", my_tools, "Get the current weather", [
    Param("location", "string", "The city and state, e.g. San Francisco, CA"),
    Param("format", "string", "The temperature unit to use. Infer this from the users location.", enum=["celsius", "fahrenheit"])
])
def get_current_weather(location, format):
    return f"It is sunny in {location} according to the weather forecast in {format}"

@tool("get_n_day_weather_forecast", my_tools, "Get an N-day weather forecast", [
    Param("location", "string", "The city and state, e.g. San Francisco, CA"),
    Param("format", "string", "The temperature unit to use. Infer this from the users location.", enum=["celsius", "fahrenheit"]),
    Param("num_days", "integer", "The number of days to forecast")
])
def get_n_day_weather_forecast(location, format, num_days):
    return f"It will be sunny for the next {num_days} days in {location} according to the weather forecast in {format}"

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
    def test_toolbox(self):
        self.assertEqual(my_tools.to_dict(), TEST_TOOLS)
        self.assertEqual(my_tools.invoke_tool(
            ToolCall(id="1", name="get_current_weather", args={"location": "San Francisco, CA", "format": "fahrenheit"})
        ), "It is sunny in San Francisco, CA according to the weather forecast in fahrenheit")

    def test_openai_tools(self):
        llm = LLM(Provider.openai, "gpt-4-turbo")
        response = llm.chat("What is the weather in San Francisco, CA in fahrenheit", tools=TEST_TOOLS)
        self.assertNotEqual(response.tools, None)
        self.assertEqual(response.tools[0].name, "get_current_weather")

        self.assertEqual(my_tools.invoke_tool(response.tools[0]), "It is sunny in San Francisco, CA according to the weather forecast in fahrenheit")

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