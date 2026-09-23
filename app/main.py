import logging
import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.api.kb import router as kb_router
from app.api.jobs import router as jobs_router
from app.api.admin import router as admin_router

# uvicorn 默认不给应用记录器配 handler,不手动加这一句,logger.info 一句都看不见
_log = logging.getLogger("app")
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())
    _log.setLevel(logging.INFO)
_log.propagate = False

_BASE_DIR = pathlib.Path(__file__).resolve().parent

app = FastAPI(title="MewHelp", version="0.1.0")
app.include_router(chat_router)
app.include_router(extract_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)

# 客服聊天页(前端):静态资源 + 首页
app.mount("/static", StaticFiles(directory=_BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_BASE_DIR / "static" / "index.html")


@app.on_event("shutdown")
def stop_owned_background_jobs() -> None:
    from app.api.jobs import get_job_runner
    get_job_runner().shutdown()
