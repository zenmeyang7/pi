from .local.providers import LocalToolProvider
from .local.search import init_retriever
from .register import ToolProvider, ToolRegister

__all__ = [
    "ToolProvider", "ToolRegister", "LocalToolProvider",
    "init_retriever", "build_registry",
]


def build_registry(mcp_config: str | None = None,
                   *, mcp_required: bool = False) -> ToolRegister:
    registry = ToolRegister()
    registry.add_provider(LocalToolProvider())

    if mcp_config:
        from .mcp.provider import McpToolProvider

        registry.add_provider(
            McpToolProvider(mcp_config, required=mcp_required))

    return registry
