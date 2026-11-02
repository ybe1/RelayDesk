"""Prompts for the structured after-sales extraction endpoint."""

from langchain_core.prompts import ChatPromptTemplate

EXTRACT_SYSTEM = """你是电商售后工单提取器。从用户的售后描述中提取结构化字段:
- order_id:订单号,仅当原文明确出现时提取,否则为 null,禁止编造或补全。千万不要传入字符串 null 或 无 这类占位文本。
- request_type:诉求类型,只能是:退款、换货、维修、投诉、其他。判断不了选「其他」。
- expected_solution:用一句话概括用户期望的处理方案,忠于原文,不添加原文没有的承诺。"""

EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_SYSTEM),
        ("human", "{text}"),
    ]
)
