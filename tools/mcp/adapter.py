import re
import json

from functools import partial
from agent.protocol import AgentTool
from .mcp_manager import McpManager
from mcp.types import TextContent

class McpToolAdapter:
    """将MCP工具适配为OpenAI兼容的Function Calling格式"""
    def __init__(self,mcp_manager:McpManager):
        self.mcp_manager = mcp_manager
        self.name_to_tool_id : dict[str,str] = {}
        self.tool_id_name:dict[str,str] = {}

    @staticmethod
    def _make_function_name(tool_id:str) -> str:
        """
        生成符合OpenAI的工具调用的约束的名称
        例如 filesystem.read_file
        ->mcp_filesystem_read_file
        """
        name = "mcp_"+tool_id.replace(".","_")
        name = re.sub(r"[^a-zA-Z0-9_-]","_",name)

        if len(name)>64:
            import hashlib

            suffix = hashlib.sha256(tool_id.encode("utf-8")).hexdigest()[:10]

            name = name[:53] + "_" + suffix

        return name

    def _flatten_result(self,result) ->str:
        parts = []
        for block in result.content or []:
            if isinstance(block,TextContent):
                parts.append(block.text)
            else:
                parts.append(f"[{getattr(block,"type","unknown")} content omitted]")
        if result.structured_content is not None and not parts:
            parts.append(json.dumps(result.structured_content,ensure_ascii=False))
        text = "\n".join(parts) or "(empty result)"
        if result.is_error:
            raise RuntimeError(text)
        return text

    def _call_tool(self,tool_id:str,arguments:dict)->str:
        return self._flatten_result(self.mcp_manager.call_tool_sync(tool_id,arguments))

    def get_openai_tools(self)->list[AgentTool]:
        tools = self.mcp_manager.get_tools()
        result = []

        self.name_to_tool_id.clear()
        self.tool_id_name.clear()

        for tool_id,meta in  tools.items():
            function_name = self._make_function_name(tool_id)
            if function_name in self.name_to_tool_id:
                raise ValueError(f"Duplicate OpenAI function name: {function_name}")
            self.name_to_tool_id[function_name] = tool_id
            self.tool_id_name[tool_id] = function_name
            definition = {
                "type":"function",
                "function":{
                    "name":function_name,
                    "description":meta.get("description", ""),
                    "parameters":meta.get("input_schema") or {
                        "type":"object",
                        "properties":{},
                    },
                }
            }
            result.append(
                AgentTool(
                    name=function_name,
                    execute=partial(self._call_tool,tool_id),
                    definition=definition,
                    execution_mode="parallel")
                )
        return result
