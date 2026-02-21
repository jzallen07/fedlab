"""Persistence models and session utilities for monitor service."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class EventRecord(Base):
    """Stored telemetry event row."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    node_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32))
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class RunStateRecord(Base):
    """Stored run state row."""

    __tablename__ = "run_state"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    state: Mapped[str] = mapped_column(String(32), default="idle")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )


def sqlite_url_from_path(path: Path) -> str:
    """Build a SQLite URL from a filesystem path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def create_session_factory(database_url: str) -> sessionmaker:
    """Create a session factory for the given database URL."""
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
