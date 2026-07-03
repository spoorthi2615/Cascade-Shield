"""
Router for graph topology endpoints.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/graph", tags=["graph"])
