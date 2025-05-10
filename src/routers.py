from fastapi import APIRouter
from .loom.router_loom import router as loom_router

router = APIRouter()

router.include_router(loom_router, prefix="/loom")
