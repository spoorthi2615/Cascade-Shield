"""
FastAPI application entrypoint for CascadeShield dashboard backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import graph, cascade, chokepoint

app = FastAPI(title="CascadeShield API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: Include routers
from backend.routers import dashboard

# app.include_router(graph.router)
# app.include_router(cascade.router)
# app.include_router(chokepoint.router)
app.include_router(dashboard.router)
