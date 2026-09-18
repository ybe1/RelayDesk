import logging

from fastapi import FastAPI

from app.api.chat import router as chat_router

# uvicorn 默认不给应用记录器配 handler,不手动加这一句,logger.info 一句都看不见
_log = logging.getLogger("app")
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())
    _log.setLevel(logging.INFO)
_log.propagate = False

app = FastAPI(title="MewHelp", version="0.1.0")
app.include_router(chat_router)
