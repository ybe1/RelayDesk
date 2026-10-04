import os
from typing import Annotated
from pydantic import Field
from mcp.server.fastmcp import FastMCP
from app.mock.business import warranty_result,return_result

mcp=FastMCP('after_sales',host='127.0.0.1',port=int(os.getenv('MCP_PORT','8102')),stateless_http=True,json_response=True)
OrderID=Annotated[str,Field(min_length=1,max_length=64)]


@mcp.tool()
def query_warranty(order_id:OrderID)->dict:
    """查询指定订单商品是否在保及演示依据。仅mock查询，不执行售后申请。"""
    return warranty_result(order_id)


@mcp.tool()
def query_return_progress(order_id:OrderID)->dict:
    """按订单号查询已有退货记录的处理进度，未查询到则明确返回空结果。仅mock数据。"""
    return return_result(order_id)


if __name__=='__main__': mcp.run(transport='streamable-http')
