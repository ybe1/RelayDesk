"""Add Ch05 persistence fields without rewriting historical data."""
import argparse
from sqlalchemy import inspect, text
from app.db.session import engine


def migrate(apply=False):
    with engine.connect() as conn:
        inspector = inspect(conn)
        columns = {c["name"]: c for c in inspector.get_columns("messages")}
        indexes = {i["name"]: i for i in inspector.get_indexes("messages")}
        if "message_key" in columns:
            col = columns["message_key"]
            if not col["nullable"] or getattr(col["type"], "length", None) != 128:
                raise RuntimeError("message_key schema mismatch; review required")
        if "uk_message_key" in indexes:
            idx = indexes["uk_message_key"]
            if not idx["unique"] or idx["column_names"] != ["message_key"]:
                raise RuntimeError("uk_message_key schema mismatch; review required")
        operations = []
        if "message_key" not in columns:
            operations.append("ALTER TABLE messages ADD COLUMN message_key VARCHAR(128) NULL")
        if "uk_message_key" not in indexes:
            operations.append("CREATE UNIQUE INDEX uk_message_key ON messages (message_key)")
        for sql in operations:
            print(("APPLY: " if apply else "PENDING: ") + sql)
            if apply:
                conn.execute(text(sql))
                conn.commit()
        if not operations:
            print("Ch05 messages schema is current")
        return operations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    migrate(parser.parse_args().apply)
