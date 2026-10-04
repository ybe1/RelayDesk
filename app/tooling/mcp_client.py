"""Refresh independent servers each turn; permission is never read from MCP hints."""
import asyncio
import logging
import httpx
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from .contracts import ToolDefinition

LOGGER=logging.getLogger(__name__)


def direct_http_client(headers=None,timeout=None,auth=None):
    return httpx.AsyncClient(headers=headers,timeout=timeout or httpx.Timeout(5,read=10),auth=auth,
                             follow_redirects=True,trust_env=False)


def transport_error(exc):
    if isinstance(exc,BaseExceptionGroup):
        errors=[transport_error(e) for e in exc.exceptions]
        return errors[0] if errors and all(errors) else None
    if isinstance(exc,(TimeoutError,httpx.TimeoutException)): return TimeoutError('MCP请求超时')
    if isinstance(exc,(ConnectionError,httpx.ConnectError,httpx.RemoteProtocolError,httpx.ReadError,httpx.WriteError)):
        return ConnectionError('MCP连接暂时不可用')
    return None


class MCPDiscovery:
    def __init__(self,connections,timeout=5):
        self.timeout=timeout
        self.connections={name:{**cfg,'httpx_client_factory':direct_http_client} for name,cfg in connections.items()}
        self.client=MultiServerMCPClient(self.connections,handle_tool_errors=False)
        self.unavailable=()

    async def discover(self):
        failed=[]
        async def one(name):
            try:
                tools=await asyncio.wait_for(self.client.get_tools(server_name=name),self.timeout)
                definitions=[]
                for tool in tools:
                    async def invoke(_tool=tool,**args):
                        try: return await _tool.ainvoke(args)
                        except Exception as exc:
                            transient=transport_error(exc)
                            if transient: raise transient from exc
                            raise
                    original=ToolDefinition.from_tool(tool,'mcp',name)
                    wrapped=StructuredTool(name=tool.name,description=tool.description,args_schema=original.parameters,coroutine=invoke)
                    definitions.append(ToolDefinition.from_tool(wrapped,'mcp',name))
                return definitions
            except Exception as exc:
                failed.append(name)
                LOGGER.warning('MCP discovery unavailable server=%s reason=%s',name,type(exc).__name__)
                return []
        groups=await asyncio.gather(*(one(name) for name in self.connections))
        self.unavailable=tuple(sorted(failed))
        return tuple(d for group in groups for d in group)
