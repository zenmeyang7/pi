import asyncio
import concurrent.futures
import json
import threading
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import Client, StdioServerParameters


class McpManager:
    def __init__(self):
        self.clients = {}
        self.tools = {}
        self.stack = AsyncExitStack()

        self._queue = None
        self._ready = threading.Event()
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mcp-loop")
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("MCP event loop startup timed out")

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    async def _serve(self):
        self._queue = asyncio.Queue()
        self._ready.set()

        while True:
            coro, future = await self._queue.get()
            if coro is None:
                break
            try:
                future.set_result(await coro)
            except Exception as e:
                future.set_exception(e)

    def _submit(self, coro):
        if self._closed:
            coro.close()
            raise RuntimeError("McpManager is closed")

        future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait, (coro, future))
        return future.result()

    def connect_server_sync(self, name, config):
        return self._submit(self.connect_server(name, config))

    def load_config_sync(self, path):
        return self._submit(self.load_config(path))

    def call_tool_sync(self, tool_id: str, arguments: dict):
        if self._closed:
            raise RuntimeError("McpManager is closed")
        return asyncio.run_coroutine_threadsafe(
            self.call_tool(tool_id, arguments), self._loop
        ).result()

    async def connect_server(self, name: str, config: dict):
        """连接mcp-server，并自动发现工具"""
        transport = config["transport"]

        if transport == "stdio":
            client = Client(
                StdioServerParameters(
                    command=config["command"],
                    args=config.get("args", []),
                    env=config.get("env"),
                )
            )
        elif transport == "streamable_http":
            client = Client(config["url"])
        else:
            raise ValueError(f"Unsupported transport: {transport}")

        client = await self.stack.enter_async_context(client)
        self.clients[name] = client

        result = await client.list_tools()

        for tool in result.tools:
            tool_id = f"{name}.{tool.name}"

            self.tools[tool_id] = {
                "server_name": name,
                "tool_name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
            }
        return result.tools

    async def load_config(self, config_path: str):
        """读取配置并连接所有server"""
        path = Path(config_path)

        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        for name, server_config in config["mcpServers"].items():
            await self.connect_server(name, server_config)

    def get_tools(self):
        """返回当前已知的工具"""
        return self.tools

    async def call_tool(self, tool_id: str, arguments: dict):
        meta = self.tools.get(tool_id)

        if meta is None:
            raise ValueError(f"Unknown tool: {tool_id}")

        client = self.clients[meta["server_name"]]
        result = await client.call_tool(
            meta["tool_name"],
            arguments=arguments,
        )
        return result

    def close_sync(self):
        if self._closed:
            return
        self._closed = True

        try:
            future = concurrent.futures.Future()
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, (self.stack.aclose(), future))
            future.result()
        finally:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, (None, None))
            self._thread.join(timeout=5)
