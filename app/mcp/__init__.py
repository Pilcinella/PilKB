"""MCP 协议适配层: 将内部 tools 包装成 MCP server 工具。"""
from app.mcp.server import build_mcp_server

__all__ = ["build_mcp_server"]
