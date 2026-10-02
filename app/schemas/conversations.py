from datetime import datetime
from pydantic import BaseModel

class ConversationItem(BaseModel):
    id:int
    created_at:datetime
    status:str
    first_question:str
    has_summary:bool

class ConversationsPage(BaseModel):
    items:list[ConversationItem]
    next_cursor:str|None

class HistoryMessage(BaseModel):
    id:int
    role:str
    content:str|None
    tool_calls:list|None
    created_at:datetime
    citations:list
    actions:list

class ConversationHistory(BaseModel):
    conversation_id:int
    messages:list[HistoryMessage]
