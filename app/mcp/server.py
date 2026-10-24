"""MCP server: exposes PilKB internal tools via Model Context Protocol.
走 stdio 传输, Claude Desktop / Cursor / Trae 这类 MCP 客户端可直连使用。
"""
from __future__ import annotations

import functools

from loguru import logger

def _register_tool(server, tool):
    """把单个 LangChain tool 挂载为 MCP tool; 拆成独立函数是为了避开循环闭包只捕获最后一个工具的坑。

    functools.wraps(tool.func) 把原函数签名复制给 wrapper: FastMCP 按签名生成
    inputSchema, 若留 (*args, **kwargs) 会被推导成 "args/kwargs 两个字符串参数",
    客户端模型就看不到真实的 query/chunk_id 等参数定义。
    """
    src = getattr(tool, "func", tool)

    @server.tool(name=tool.name, description=tool.description)
    @functools.wraps(src)
    def tool_wrapper(*args, **kwargs):
        try:
            return tool.invoke(kwargs or (args[0] if args else {}))
        except Exception as e:
            logger.exception(f"MCP tool {tool.name} failed: {e}")
            return f"Error: {e}"

    return tool_wrapper


def build_mcp_server(kb_id: str = "default"):
    """构造 MCP server 实例, 工具集合即内部的 hybrid_search / calculator / current_date。"""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        logger.error("mcp not installed. `pip install mcp`")
        raise

    # 延迟导入: 只用工具适配函数时, 不必强装整套 Agent 依赖。
    from app.agents.tools import build_tools

    server = FastMCP("PilKB")

    tools = build_tools(kb_id)

    for tool in tools:
        _register_tool(server, tool)

    return server


def main():
    """stdio 模式入口: python -m app.mcp.server"""
    import sys
    kb_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    logger.info(f"Starting PilKB MCP server (stdio) for kb={kb_id}")
    server = build_mcp_server(kb_id)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
