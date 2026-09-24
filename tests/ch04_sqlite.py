"""SQLite DDL adapters for MySQL-specific Ch04 ORM column types in tests."""

from sqlalchemy.dialects import mysql
from sqlalchemy.ext.compiler import compiles


@compiles(mysql.ENUM, "sqlite")
def compile_mysql_enum(element, compiler, **kw):
    return "VARCHAR(64)"


@compiles(mysql.BIGINT, "sqlite")
def compile_mysql_bigint(element, compiler, **kw):
    return "INTEGER" if kw.get("type_expression", None) is not None and kw["type_expression"].primary_key else "BIGINT"
