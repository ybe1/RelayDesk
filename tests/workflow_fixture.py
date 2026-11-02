"""Current chat workflow with in-memory persistence for offline HTTP tests."""

from app.workflow.graph import build_workflow
from app.workflow.service import WorkflowChatService


class MemoryStore:
    session_factory = None

    def __init__(self):
        self.rows = []

    def start_or_load(self, user_id, conversation_id):
        return conversation_id or 7

    def load_history(self, conversation_id):
        return list(self.rows)


class OfflineWorkflowService(WorkflowChatService):
    def __init__(self, model, **kwargs):
        self.gaps = []
        super().__init__(build_workflow(), MemoryStore(), model_factory=lambda: model,
                         tools_factory=lambda *args, **kwargs: [], **kwargs)

    def _save(self, cid, turn_id, message):
        if message.type == "tool":
            return None
        self.store.rows.append(message)
        return len(self.store.rows)

    def _pool(self, cid, raw, source, reason, snapshot=None, event_key=None):
        self.gaps.append({"raw": raw, "source": source, "reason": reason})

    def _offers(self, cid, turn_id, raw, actions, refund_defaults=None):
        return [{"action": action} for action in actions]
