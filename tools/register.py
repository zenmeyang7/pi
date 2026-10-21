import logging
from typing import Protocol,List
from agent.protocol import AgentTool

log = logging.getLogger(__name__)

class ToolProvider(Protocol):
    name:str
    def list_tools(self)->list[AgentTool]:
        """返回当前可用工具"""

    def close(self) -> None:
        """释放资源"""

class ToolRegister:
    def __init__(self):
        self._providers:list[ToolProvider] = []
        self._tools:dict[str,AgentTool] = {}

    def add_provider(self,provider:ToolProvider) -> "ToolRegister":
        tools = provider.list_tools()

        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(
                    f"Duplicate tool name {tool.name}"
                    f"(provider={provider.name})"
                )

        for tool in tools:
            self._tools[tool.name] = tool
        self._providers.append(provider)
        return self

    def list_tools(self)->List[AgentTool]:
        return list(self._tools.values())

    def get(self,name:str) -> AgentTool|None:
        return self._tools.get(name)

    def close(self)->None:
        for provider in reversed(self._providers):
            try:
                provider.close()
            except Exception:
                log.exception(f"failed to close tool provider{provider.name}")
        self._providers.clear()
        self._tools.clear()