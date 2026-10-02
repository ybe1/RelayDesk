"""Read-only, owned conversation listing and original-message history."""
import asyncio
from typing import Annotated
from fastapi import APIRouter,Request,Query,Path,HTTPException,Depends
from app.db.session import SessionLocal
from app.repositories.conversations import list_owned,owned_history
from app.schemas.conversations import ConversationsPage,ConversationHistory

router=APIRouter()

def get_conversation_sessions(): return SessionLocal

@router.get('/api/conversations',response_model=ConversationsPage)
async def conversations(user_id:Annotated[str,Query(min_length=1,max_length=64)],
                        cursor:Annotated[str|None,Query(max_length=256)]=None,
                        limit:Annotated[int,Query(ge=1,le=100)]=50,
                        factory=Depends(get_conversation_sessions)):
    def read():
        with factory() as session: return list_owned(session,user_id,cursor,limit)
    try: return await asyncio.to_thread(read)
    except ValueError as exc: raise HTTPException(422,'无效会话游标') from exc

@router.get('/api/conversations/{conversation_id}/messages',response_model=ConversationHistory)
async def messages(request:Request,conversation_id:Annotated[int,Path(gt=0)],
                   user_id:Annotated[str,Query(min_length=1,max_length=64)],
                   factory=Depends(get_conversation_sessions)):
    def read():
        with factory() as session: return owned_history(session,user_id,conversation_id)
    try: result=await asyncio.to_thread(read)
    except PermissionError as exc: raise HTTPException(403,'会话不属于当前用户') from exc
    except LookupError as exc: raise HTTPException(404,'会话不存在') from exc
    service=getattr(request.app.state,'workflow_service',None)
    if service and getattr(service.graph,'checkpointer',None):
        snapshot=await service.graph.aget_state(service._config(conversation_id))
        metadata={m.id:m.additional_kwargs for m in snapshot.values.get('messages',[])}
        for row in result['messages']:
            row['citations']=metadata.get(row['message_key'],{}).get('citations',[])
    return result
