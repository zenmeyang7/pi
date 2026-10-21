from typing import List

from agent.protocol import AgentTool

from ..register import ToolProvider
from . import files, search


class LocalToolProvider(ToolProvider):
    name = "local"

    def __init__(self):
        self._tools = [files.read_file(), files.write_file(), search.search_code()]

    def list_tools(self) -> List[AgentTool]:
        return list(self._tools)

    def close(self) -> None:
        pass
