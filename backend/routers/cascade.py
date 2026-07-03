"""
Router for cascade simulation and propagation endpoints.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/cascade", tags=["cascade"])
