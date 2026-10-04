"""Independent, bounded, best-effort audit persistence."""
import logging
from functools import lru_cache
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import ToolAuditLog

LOGGER=logging.getLogger(__name__)


@lru_cache(maxsize=1)
def audit_session_factory():
    from app.config import settings
    engine=create_engine(settings.database_url,pool_size=2,max_overflow=0,pool_timeout=1,pool_pre_ping=True,
                         connect_args={'connect_timeout':1,'read_timeout':1,'write_timeout':1})
    return sessionmaker(bind=engine,expire_on_commit=False)


class ToolAuditRepository:
    def __init__(self,session_factory=None): self.session_factory=session_factory or audit_session_factory()
    def write(self,record):
        try:
            with self.session_factory() as session,session.begin():
                session.add(ToolAuditLog(**record))
            return True
        except Exception:
            LOGGER.exception('tool audit write failed tool=%s call_id=%s',record.get('tool_name'),record.get('tool_call_id'))
            return False
