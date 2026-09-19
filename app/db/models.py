from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text, text
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_user_id", "user_id"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(mysql.ENUM("进行中", "已转人工", "已结束"), nullable=False, server_default=text("'进行中'"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_messages_conversation"), nullable=False)
    role: Mapped[str] = mapped_column(mysql.ENUM("user", "assistant", "tool"), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class FAQ(Base):
    __tablename__ = "faq"
    __table_args__ = (Index("idx_category", "category"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_tickets_conversation"), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(mysql.ENUM("售后", "投诉", "咨询"), nullable=False)
    status: Mapped[str] = mapped_column(mysql.ENUM("待处理", "已处理"), nullable=False, server_default=text("'待处理'"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
