from typing import Any
from muxllm.providers.base import ToolCall

class ToolBox:
    def __init__(self):
        self.tools = {}

    def add_tool(self, tool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str):
        return self.tools.get(name)
    
    def invoke_tool(self, tool_call: ToolCall):
        tool = self.get_tool(tool_call.name)
        if tool:
            return tool(**tool_call.args)
        else:
            return None
        
    def to_dict(self) -> dict[str, Any]:
        return [tool.to_dict() for tool in self.tools.values()]

class Param:
    def __init__(self, name: str, type: str, description: str, enum: list[str] = None):
        self.name = name
        self.description = description
        self.type = type
        self.enum = enum

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": self.type,
            "description": self.description
        }

        if self.enum:
            d["enum"] = self.enum

        return d

class Tool:
    def __init__(self, name: str, description: str, parameters: dict, function: callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function" : {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {param.name: param.to_dict() for param in self.parameters},
                    "required": [param.name for param in self.parameters]
                }
            }
        }
    
    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

def tool(name: str,
         toolBox: ToolBox,
         description: str,
         parameters: list[Param]):
    def decorator(func):
        tool = Tool(name, description, parameters, func)
        toolBox.add_tool(tool)
        return func
    return decorator