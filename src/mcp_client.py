import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_mcp_tool(tool_name: str, arguments: dict | None = None):
    """
    Connect to the Enterprise Knowledge MCP server
    and execute a tool.
    """

    server_path = Path("src/mcp_server/server.py").resolve()

    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "python",
            str(server_path),
        ],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments or {},
            )

            return result


def call_mcp(tool_name: str, arguments: dict | None = None):
    """
    Synchronous wrapper around the async MCP client.
    """

    return asyncio.run(
        call_mcp_tool(tool_name, arguments)
    )