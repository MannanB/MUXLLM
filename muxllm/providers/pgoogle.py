import google.generativeai as genai
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
            return {"role": "model",
                    "parts": [response.message] if response.message else []}
        else:
            return {"role": "model",
                    "parts": [{"functionCall": {
                        "name": tool.name,
                        "args": tool.args
                        }} for tool in response.tools]
                    }

    def parse_tool_response(self, tool_resp: ToolResponse) -> dict:
        return {
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": tool_resp.name,
                    "response": {
                        "name": tool_resp.name,
                        "content": tool_resp.content
                    }
            }}]
        }

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

        if messages[0]["role"] == "system":
            system_message = messages[0]["parts"][0]
            client = genai.GenerativeModel(model, system_instruction=system_message, tools=google_proto_tools)
            messages = messages[1:]
        else:
            client = genai.GenerativeModel(model)

        response = client.generate_content(messages)

        tools = []
        for part in response.candidates[0].content:
            if fn := part.function_call:
                tools.append(ToolCall(id='', name=fn.name, args={k: v for k, v in fn.args.items()}))

        return LLMResponse(model=model, raw_response=response, message=response.text, tools=tools)
    
    # async def get_response_async(self, messages : list[dict[str, str | dict]], model : str, **kwargs) -> LLMResponse:
    #     model = self.validate_model(model)

    #     response = await self.async_client.messages.create(
    #                         model=model,
    #                         messages=messages,
    #                         **kwargs) 
        
    #     if response.stop_reason == "tool_use":
    #         tool_uses = [block for block in response.content if block.type == "tool_use"]
    #         thinking = next(block for block in response.content if block.type == "text")
    #         return LLMResponse(model=model, raw_response=response, message=thinking.text, tools=[ToolCall(id=tool_use.id, name=tool_use.name, args=tool_use.input) for tool_use in tool_uses])
    #     return LLMResponse(model=model, raw_response=response, message=response.content.text, tools=None)