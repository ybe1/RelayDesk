import os
from typing import Annotated
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from app.mock.business import logistics_result

mcp=FastMCP('logistics',host='127.0.0.1',port=int(os.getenv('MCP_PORT','8101')),stateless_http=True,json_response=True)


@mcp.tool()
def query_logistics(order_id:Annotated[str,Field(min_length=1,max_length=64)])->dict:
    """按订单号查询物流状态、位置及轨迹。返回mock演示数据，不代表真实配送。"""
    return logistics_result(order_id)


if __name__=='__main__': mcp.run(transport='streamable-http')
