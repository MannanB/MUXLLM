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
            api_key = os.getenv("GEMINI_API_KEY")

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
            return {"role": "assistant",
                    "content": response.message if response.message else ""}
        else:
            return {"role": "assistant",
                    "content": [{"type": " text",
                                 "text" : response.message if response.message else ""}] + 
                               [{"type": "tool_use",
                                 "id": tool.id,
                                 "name": tool.name,
                                 "input": tool.args
                               } for tool in response.tools]}  

    # def parse_tool_response(self, tool_resp: ToolResponse) -> dict:
    #     return {
    #         "role": "user",
    #         "content": {
    #             "type": "tool_result",
    #             "tool_use_id": tool_resp.id,
    #             "content": tool_resp.response
    #         }
    #     }

    def get_response(self, messages : list[dict[str, str | dict]], model : str, **kwargs) -> LLMResponse:
        model = self.validate_model(model)

        if messages[0]["role"] == "system":
            system_message = messages[0]["parts"][0]
            client = genai.GenerativeModel(model, system_instruction=system_message)
            messages = messages[1:]
        else:
            client = genai.GenerativeModel(model)

        response = client.generate_content(messages)

        # if response.stop_reason == "tool_use":
        #     tool_uses = [block for block in response.content if block.type == "tool_use"]
        #     thinking = next(block for block in response.content if block.type == "text")
        #     return LLMResponse(model=model, raw_response=response, message=thinking.text, tools=[ToolCall(id=tool_use.id, name=tool_use.name, args=tool_use.input) for tool_use in tool_uses])
        return LLMResponse(model=model, raw_response=response, message=response.candidates[0].content, tools=None)
    
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