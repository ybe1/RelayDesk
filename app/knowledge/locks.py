from contextlib import contextmanager

from sqlalchemy import text


class WriterBusy(RuntimeError):
    pass


@contextmanager
def writer_lock(engine):
    """Hold the MySQL connection-level writer lock for one write operation."""
    connection = engine.connect()
    acquired = False
    try:
        acquired = connection.execute(text("SELECT GET_LOCK('mewhelp_ch03_writer', 0)")).scalar() == 1
        if not acquired:
            raise WriterBusy("another knowledge writer is active")
        yield
    finally:
        if acquired:
            connection.execute(text("SELECT RELEASE_LOCK('mewhelp_ch03_writer')"))
        connection.close()
