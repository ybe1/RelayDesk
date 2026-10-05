from pathlib import Path
from app.config import settings
from .builtins import bind_builtins
from .registry import ToolRegistry
from .policy import PolicyStore
from .mcp_client import MCPDiscovery


def create_runtime_tools():
    root=Path(__file__).resolve().parents[2]
    registry=ToolRegistry(PolicyStore(root/settings.tool_policy_path),root/settings.tool_plugins_path)
    registry.register_builtin(bind_builtins)
    discovery=MCPDiscovery({
        'logistics':{'url':settings.mcp_logistics_url,'transport':'streamable_http'},
        'after_sales':{'url':settings.mcp_after_sales_url,'transport':'streamable_http'}})
    return registry,discovery
