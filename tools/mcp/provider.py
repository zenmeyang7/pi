import logging
from typing import List

from agent.protocol import AgentTool

from ..register import ToolProvider
from .adapter import McpToolAdapter
from .mcp_manager import McpManager

log = logging.getLogger(__name__)


class McpToolProvider(ToolProvider):
    name = "mcp"

    def __init__(self, config_path: str, required: bool = False):
        self._manager: McpManager | None = None
        self._tools: List[AgentTool] = []

        try:
            self._manager = McpManager()
            self._manager.load_config_sync(config_path)
            self._tools = McpToolAdapter(self._manager).get_openai_tools()
        except Exception:
            if required:
                self.close()
                raise
            log.exception("MCP provider unavailable")

    def list_tools(self) -> List[AgentTool]:
        return list(self._tools)

    def close(self) -> None:
        if self._manager is None:
            return
        manager, self._manager = self._manager, None
        manager.close_sync()
