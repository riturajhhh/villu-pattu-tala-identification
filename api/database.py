"""
api/database.py
================
SQLite database access and ORM setup for logging prediction history.
"""

from __future__ import annotations

import datetime
import os
import sys
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import get_config

# Resolve database URL
try:
    cfg = get_config()
    db_path = Path(cfg.api.history_db)
except Exception:
    db_path = Path("outputs/history.db")

db_path.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{db_path.absolute()}"

from sqlalchemy import create_engine

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class PredictionHistory(Base):
    """ORM Model for tracking past API inferences."""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    predicted_tala = Column(String, index=True)
    confidence = Column(Float)
    bpm = Column(Float)
    pulse_clarity = Column(Float)
    duration = Column(Float)
    model_used = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


def init_db() -> None:
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
