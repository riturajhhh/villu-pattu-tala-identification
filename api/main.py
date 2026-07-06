"""
api/main.py
============
FastAPI entry point for the Villu Pattu Tala Identification System.
Registers routers, databases, CORS, and runs uvicorn server.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.database import init_db
from api.routes import features, history, model_info, predict, upload
from utils.config_loader import get_config
from utils.logger import init_from_config

# Initialise logging
logger = init_from_config()

# Setup database tables
try:
    init_db()
    logger.info("Database tables initialised successfully.")
except Exception as exc:
    logger.critical(f"Failed to initialise database: {exc}")

app = FastAPI(
    title="Villu Pattu Tala Identification API",
    description="Research-ready API for identifying rhythmic cycles (Talas) of Villu Pattu performances.",
    version="1.0.0",
)

# CORS Middleware setup
try:
    cfg = get_config()
    origins = list(cfg.api.cors_origins)
except Exception:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route sub-modules
app.include_router(predict.router, tags=["Prediction"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(history.router, tags=["History"])
app.include_router(features.router, tags=["Features"])
app.include_router(model_info.router, tags=["Model Info"])


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "project": "Villu Pattu Tala Identification System",
        "api_version": "1.0.0"
    }


if __name__ == "__main__":
    try:
        cfg = get_config()
        host = cfg.api.host
        port = int(cfg.api.port)
    except Exception:
        host = "0.0.0.0"
        port = 8000

    uvicorn.run("api.main:app", host=host, port=port, reload=True)
