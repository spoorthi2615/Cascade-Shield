"""
Router for chokepoint ranking and intervention endpoints.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/chokepoint", tags=["chokepoint"])
