"""SSE entry point for the customer service tool chain."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse

from app.core.llm import get_chat_model
from app.db.session import SessionLocal
from app.schemas.chat import ChatRequest,SelectionRequest
from app.services.chat import ChatService, DatabaseChatStore
from app.services.tool_executor import ToolExecutor
from app.tools.business import build_tools
from app.context.budget import ContextCapacityError

logger = logging.getLogger(__name__)
router = APIRouter()


def get_chat_service(request: Request):
    return request.app.state.workflow_service


@router.post("/api/chat")
async def chat(req: ChatRequest, service: ChatService = Depends(get_chat_service)):
    from app.workflow.service import WorkflowChatService
    prepared = None
    if isinstance(service, WorkflowChatService):
        try:
            prepared = await service.prepare(req)
        except PermissionError as exc:
            raise HTTPException(403, "会话不属于当前用户") from exc
        except LookupError as exc:
            raise HTTPException(404, "会话不存在") from exc
        except RuntimeError as exc:
            raise HTTPException(409, "此会话正在处理，请等待本轮完成") from exc
    return stream_chat(req,service,prepared)


def stream_chat(req,service,prepared,selection=False):
    async def event_stream() -> AsyncIterator[str]:
        events = None
        try:
            events = service.selection_events(req,prepared=prepared) if selection else (service.events(req, prepared=prepared) if prepared else service.events(req))
            async for kind, data in events:
                if kind == "done":
                    yield "data: [DONE]\n\n"
                elif kind == "delta":
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                else:
                    yield f"event: {kind}\n"
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except PermissionError:
            yield "event: error\n"
            yield f"data: {json.dumps({'message': '会话不属于当前用户'}, ensure_ascii=False)}\n\n"
        except LookupError:
            yield "event: error\n"
            yield f"data: {json.dumps({'message': '会话不存在'}, ensure_ascii=False)}\n\n"
        except ContextCapacityError as exc:
            yield "event: error\n"
            yield f"data: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("聊天工具链失败 conversation_id=%s", req.conversation_id)
            yield "event: error\n"
            yield f"data: {json.dumps({'message': '服务暂时不可用，请稍后重试'}, ensure_ascii=False)}\n\n"
        finally:
            if events is not None:
                await events.aclose()
            if prepared is not None:
                service.release(prepared)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post('/api/chat/selection')
async def select_order(req:SelectionRequest,service=Depends(get_chat_service)):
    try: prepared=await service.prepare_selection(req)
    except PermissionError as exc: raise HTTPException(403,'会话不属于当前用户') from exc
    except LookupError as exc: raise HTTPException(404,'会话不存在') from exc
    except RuntimeError as exc: raise HTTPException(409,'选单已失效或会话正在处理') from exc
    except ValueError as exc: raise HTTPException(422,'订单不在候选列表') from exc
    return stream_chat(req,service,prepared,selection=True)


@router.get('/api/chat/{conversation_id}/pending')
async def pending_order(conversation_id:int,user_id:str,service=Depends(get_chat_service)):
    try: pending=await service.pending_selection(user_id,conversation_id)
    except PermissionError as exc: raise HTTPException(403,'会话不属于当前用户') from exc
    except LookupError as exc: raise HTTPException(404,'会话不存在') from exc
    return {'pending':pending}
