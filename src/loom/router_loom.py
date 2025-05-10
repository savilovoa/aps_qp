from fastapi import APIRouter
from .model_loom import DataLoomIn, LoomPlansOut
from starlette.responses import Response
from .schedule_loom import schedule_loom_calc

router = APIRouter()

@router.post("/plan", response_model=LoomPlansOut)
async def calc_loom_plan(plan_in: DataLoomIn, response: Response):
    plan_out = schedule_loom_calc(plan_in)
    if plan_out.error_str != "":
        response.status_code = 500
    return plan_out

