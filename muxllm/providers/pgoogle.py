import google.generativeai as genai
import proto
import os
from muxllm.providers.base import CloudProvider, LLMResponse, ToolCall, ToolResponse
from typing import Optional

model_alias = {}

available_models = []

class GoogleProvider(CloudProvider):
    def __init__(self, api_key : Optional[str] = None):
        super().__init__(available_models, model_alias)
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")

        genai.configure(api_key=api_key)

    def parse_system_message(self, message: str) -> dict:
        # TODO: 
        return {
            "role": "system",
            "parts": [message]
        }

    def parse_user_message(self, message: str) -> dict:
        return {
            "role": "user",
            "parts": [message]
        }

    def parse_response(self, response: LLMResponse) -> dict:
        if not response.tools:
            resp = {"role": "model",
                    "parts": [genai.protos.Part({"text": response.message})] if response.message else []}
        else:
            resp = {"role": "model",
                    "parts": [genai.protos.Part({"function_call": genai.protos.FunctionCall({  
                        "name": tool.name,
                        "args": tool.args
                        })}) for tool in response.tools]
                    }
        return genai.protos.Content(resp)



    def parse_tool_response(self, tool_resp: ToolResponse) -> dict:
        return genai.protos.Content({
            "role": "function",
            "parts": [{
                "function_response": genai.protos.FunctionResponse({
                    "name": tool_resp.name,
                    "response": {
                        "name": tool_resp.name,
                        "content": tool_resp.response
                    }
            })}]
        })

    def tools_dict_to_google_protos(self, tools: list[dict[str, str | dict]]) -> list[genai.protos.Tool]:
        google_proto_tools = []
        for tool in tools:
            google_proto_tool = {
                'function_declarations': [
                    {
                        'name': tool['function']['name'],
                        'description': tool['function']['description'],
                        'parameters': {
                            'type_': tool['function']['parameters']['type'].upper(),
                            'properties': {
                                prop_name: {'type_': prop_data['type'].upper(), "description": prop_data['description']} for prop_name, prop_data in tool['function']['parameters']['properties'].items()
                            },
                            'required': tool['function']['parameters']['required']
                        }
                    }
                ]
            }
            google_proto_tools.append(genai.protos.Tool(google_proto_tool))
        return google_proto_tools

    def get_response(self, messages : list[dict[str, str | dict]], model : str, **kwargs) -> LLMResponse:
        model = self.validate_model(model)

        google_proto_tools = []
        if "tools" in kwargs:
            google_proto_tools = self.tools_dict_to_google_protos(kwargs["tools"])
        # google doesnt need tool_choice, delete it if it exists
        if "tool_choice" in kwargs:
            del kwargs["tool_choice"]

        if messages[0]["role"] == "system":
            system_message = messages[0]["parts"][0]
            client = genai.GenerativeModel(model, system_instruction=system_message, tools=google_proto_tools)
            messages = messages[1:]
        else:
            client = genai.GenerativeModel(model, tools=google_proto_tools)

        response = client.generate_content(messages)

        
        tools = []
        for part in response.candidates[0].content.parts:
            if fn := part.function_call:
                tools.append(ToolCall(id='', name=fn.name, args={k: v for k, v in fn.args.items()}))
        if tools:
            return LLMResponse(model=model, raw_response=response, message="", tools=tools)
        else:
            return LLMResponse(model=model, raw_response=response, message=response.text, tools=None)
    
    async def get_response_async(self, messages : list[dict[str, str | dict]], model : str, **kwargs) -> LLMResponse:
        model = self.validate_model(model)

        google_proto_tools = []
        if "tools" in kwargs:
            google_proto_tools = self.tools_dict_to_google_protos(kwargs["tools"])

        if messages[0]["role"] == "system":
            system_message = messages[0]["parts"][0]
            client = genai.GenerativeModel(model, system_instruction=system_message, tools=google_proto_tools)
            messages = messages[1:]
        else:
            client = genai.GenerativeModel(model, tools=google_proto_tools)

        response = await client.generate_content_async(messages)
        
        tools = []
        for part in response.candidates[0].content.parts:
            if fn := part.function_call:
                tools.append(ToolCall(id='', name=fn.name, args={k: v for k, v in fn.args.items()}))
        if tools:
            return LLMResponse(model=model, raw_response=response, message="", tools=tools)
        else:
            return LLMResponse(model=model, raw_response=response, message=response.text, tools=None)