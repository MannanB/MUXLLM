import os
from typing import Optional
from muxllm.providers.base import CloudProvider, LLMResponse, ToolCall, ToolResponse
import anthropic

model_alias = {
    "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
}

class AnthropicProvider(CloudProvider):
    def __init__(self, api_key : Optional[str] = None):
        super().__init__(model_alias)
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)

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

    def parse_tool_response(self, tool_resp: ToolResponse) -> dict:
        return {
            "role": "user",
            "content": {
                "type": "tool_result",
                "tool_use_id": tool_resp.id,
                "content": tool_resp.response
            }
        }

    def get_response(self, messages : list[dict[str, str | dict]], model : str, **kwargs) -> LLMResponse:
        model = self.validate_model(model)
        
        response = self.client.messages.create(
                    model=model,
                    messages=messages,
                    **kwargs) 
        
        if response.stop_reason == "tool_use":
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            thinking = next(block for block in response.content if block.type == "text")
            return LLMResponse(model=model, raw_response=dict(response), message=thinking.text, tools=[ToolCall(id=tool_use.id, name=tool_use.name, args=tool_use.input) for tool_use in tool_uses])
        return LLMResponse(model=model, raw_response=response, message=response.content.text, tools=None)
    
    async def get_response_async(self, messages : list[dict[str, str | dict]], model : str, **kwargs) -> LLMResponse:
        model = self.validate_model(model)

        response = await self.async_client.messages.create(
                            model=model,
                            messages=messages,
                            **kwargs) 
        
        if response.stop_reason == "tool_use":
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            thinking = next(block for block in response.content if block.type == "text")
            return LLMResponse(model=model, raw_response=dict(response), message=thinking.text, tools=[ToolCall(id=tool_use.id, name=tool_use.name, args=tool_use.input) for tool_use in tool_uses])
        return LLMResponse(model=model, raw_response=response, message=response.content.text, tools=None)
    
    
    # def get_response_stream(self, messages : list[dict[str, str]], model : str, **kwargs):
    #     model = self.validate_model(model)
        
    #     response = self.client.messages.stream(
    #                 model=model,
    #                 messages=messages,
    #                 **kwargs) 
        
    #     for chunk in response:
    #         yield LLMResponse(model=model, raw_response=chunk, message=chunk.content.text, tools=chunk.choices[0].message.tools)