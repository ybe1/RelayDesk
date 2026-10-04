import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import pytest


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); return sock.getsockname()[1]


@pytest.fixture
async def mcp_pair(tmp_path):
    from app.tooling.mcp_client import MCPDiscovery
    root=Path(__file__).resolve().parents[1]
    ports={'logistics':free_port(),'after_sales':free_port()}; processes={}; handles=[]
    async def start(name,extra=False):
        if name in processes:
            processes[name].terminate(); await asyncio.to_thread(processes[name].wait,timeout=5)
        script=tmp_path/(name+'.py')
        source=f'from mcp_servers.{name} import mcp\n'
        if extra: source+='@mcp.tool()\ndef query_delivery_note(order_id: str) -> dict:\n    """Query delivery note."""\n    return {"order_id":order_id,"note":"演示配送说明","mock":True}\n'
        source+='mcp.run(transport="streamable-http")\n'
        script.write_text(source,encoding='utf-8')
        env={**os.environ,'MCP_PORT':str(ports[name]),'PYTHONPATH':str(root),'PYTHONIOENCODING':'utf-8'}
        handle=(tmp_path/(name+'.log')).open('ab'); handles.append(handle)
        processes[name]=subprocess.Popen([sys.executable,str(script)],cwd=root,env=env,stdout=handle,stderr=handle,
                                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        for _ in range(100):
            try:
                reader,writer=await asyncio.open_connection('127.0.0.1',ports[name]); writer.close(); await writer.wait_closed(); return
            except OSError:
                if processes[name].poll() is not None: raise AssertionError((tmp_path/(name+'.log')).read_text(errors='replace'))
                await asyncio.sleep(.1)
        raise AssertionError('MCP server did not start')
    discovery=MCPDiscovery({n:{'url':f'http://127.0.0.1:{p}/mcp','transport':'streamable_http'} for n,p in ports.items()})
    try:
        await start('logistics'); await start('after_sales')
        yield SimpleNamespace(discovery=discovery,restart=start,processes=processes)
    finally:
        for process in processes.values():
            if process.poll() is None: process.terminate()
            await asyncio.to_thread(process.wait,timeout=5)
        for handle in handles: handle.close()


async def test_http_calls_and_consistent_mock(mcp_pair):
    from app.tooling.results import normalize_result
    from app.mock.business import mock_order_snapshot
    definitions=await mcp_pair.discovery.discover()
    assert {d.mcp_server for d in definitions}=={'logistics','after_sales'}
    for d in definitions:
        result=normalize_result(d,await d.tool.ainvoke({'order_id':'1001'}))
        assert result['result']['mock'] is True
        if d.name=='query_logistics':
            assert result['result']['status']==mock_order_snapshot('1001')['logistics_status']
            assert isinstance(result['result']['tracks'],list)


async def test_server_only_restart_hot_discovers_new_tool_and_isolates_failure(mcp_pair):
    before=await mcp_pair.discovery.discover()
    assert 'query_delivery_note' not in {d.name for d in before}
    await mcp_pair.restart('logistics',extra=True)
    after=await mcp_pair.discovery.discover()
    added=next(d for d in after if d.name=='query_delivery_note')
    assert added.mcp_server=='logistics'
    assert await added.tool.ainvoke({'order_id':'1001'})
    mcp_pair.processes['after_sales'].terminate()
    await asyncio.to_thread(mcp_pair.processes['after_sales'].wait,timeout=5)
    remaining=await mcp_pair.discovery.discover()
    assert remaining and {d.mcp_server for d in remaining}=={'logistics'}


def test_runtime_builtin_catalog_has_no_logistics():
    from app.tooling.builtins import bind_builtins
    from app.tooling.contracts import CallContext
    tools=bind_builtins(CallContext(conversation_id=1,session_factory=lambda:None))
    assert {t.name for t in tools}=={'query_order','query_product','query_faq','create_ticket'}
