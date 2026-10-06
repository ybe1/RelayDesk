from langchain_core.prompts import ChatPromptTemplate


FLYWHEEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你整理客服知识缺口。把原问题改写成独立、简洁的 FAQ 问句，并给一条仅供审核的示例答案。
判断它是否与候选问题语义相同；型号、否定、期限、业务动作不同就不能合并。
命中时 matched_question_id 必须取候选 ID，否则为 null。只输出规定结构，不编造用户未提供的事实。"""),
    ("human", "{payload}"),
])
