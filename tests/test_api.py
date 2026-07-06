"""
tests/test_api.py
=================
API endpoint route testing using FastAPI TestClient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app


def test_api_health_check():
    """Test health check health endpoint."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "project" in data


def test_api_model_info():
    """Test model-info endpoint returns expected classes."""
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "classes" in data
        assert isinstance(data["classes"], list)


def test_api_history():
    """Test history logs query returns records list."""
    with TestClient(app) as client:
        response = client.get("/history?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
