import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPClient:
    def __init__(self, command: List[str]):
        if not command:
            raise ValueError("command is required")
        self.command = list(command)

    def list_tools(self):
        return asyncio.run(self._list_tools())

    def call_tool(self, name, arguments=None):
        return asyncio.run(self._call_tool(name, arguments or {}))

    def _server_parameters(self):
        return StdioServerParameters(command=self.command[0], args=self.command[1:])

    async def _list_tools(self):
        async with Client(stdio_client(self._server_parameters())) as client:
            result = await client.list_tools()
            return [MCPTool(tool.name, tool.description or "", tool.input_schema or {}) for tool in result.tools]

    async def _call_tool(self, name, arguments):
        async with Client(stdio_client(self._server_parameters())) as client:
            return await client.call_tool(name, arguments)
