import logging
import pathlib
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.api.kb import router as kb_router
from app.api.jobs import router as jobs_router
from app.api.admin import router as admin_router
from app.api.rageval import router as rageval_router
from app.api.chat_actions import router as chat_actions_router
from app.api.refunds import router as refunds_router
from app.api.conversations import router as conversations_router

# uvicorn 默认不给应用记录器配 handler,不手动加这一句,logger.info 一句都看不见
_log = logging.getLogger("app")
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())
    _log.setLevel(logging.INFO)
_log.propagate = False
_log_dir=pathlib.Path(__file__).resolve().parent.parent/'log'
_log_dir.mkdir(exist_ok=True)
if not any(isinstance(h,logging.FileHandler) and pathlib.Path(h.baseFilename)==(_log_dir/'app.log') for h in _log.handlers):
    handler=logging.FileHandler(_log_dir/'app.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    _log.addHandler(handler)

_BASE_DIR = pathlib.Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(application):
    from app.config import settings
    from app.workflow.runtime import checkpoint_runtime
    from app.workflow.graph import build_workflow
    from app.workflow.service import WorkflowChatService
    from app.services.chat import DatabaseChatStore
    from app.db.session import SessionLocal
    from app.api.jobs import get_job_runner
    from app.context.budget import config_from_settings,validate_context_capacity
    from app.context.compile import AGENT_SYSTEM
    from app.context.tokens import estimate_messages
    from app.tools.business import build_tools
    from app.workflow.contracts import AgentDecision
    from langchain_core.messages import SystemMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool
    cfg=config_from_settings(settings)
    schemas=[convert_to_openai_tool(t) for t in [*[t for t in build_tools(SessionLocal,0) if t.name!='create_ticket'],AgentDecision]]
    breakdown=validate_context_capacity(cfg,estimate_messages([SystemMessage(content=AGENT_SYSTEM)],schemas))
    _log.info('context startup fixed_prefix=%s history=%s layer1=%s layer2=%s',estimate_messages([SystemMessage(content=AGENT_SYSTEM)],schemas),breakdown.history,breakdown.layer1,breakdown.layer2)
    async with checkpoint_runtime(settings.workflow_checkpoint_path) as saver:
        application.state.workflow_service = WorkflowChatService(build_workflow(checkpointer=saver), DatabaseChatStore(SessionLocal))
        try:
            yield
        finally:
            await application.state.workflow_service.summary_worker.aclose()
            await asyncio.to_thread(get_job_runner().shutdown)
            application.state.workflow_service = None


app = FastAPI(title="MewHelp", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)
app.include_router(rageval_router)
app.include_router(chat_actions_router)
app.include_router(refunds_router)
app.include_router(conversations_router)

# 客服聊天页(前端):静态资源 + 首页
app.mount("/static", StaticFiles(directory=_BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "index.html")


@app.get("/kb", include_in_schema=False)
async def knowledge_base_page() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "kb.html")


@app.get("/admin", include_in_schema=False)
async def admin_page() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "admin.html")


@app.get("/rag-eval", include_in_schema=False)
async def rag_eval_page() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "rag-eval.html")


