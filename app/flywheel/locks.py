from __future__ import annotations

import threading


FLYWHEEL_LOCK_NAME = "mewhelp:ch09:flywheel"
LOCAL_FLYWHEEL_LOCK = threading.Lock()
