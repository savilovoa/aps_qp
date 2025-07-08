from fastapi import APIRouter
from .model_loom import DataLoomIn, LoomPlansOut, LoomPlansViewIn, LoomPlansViewOut
from starlette.responses import Response
from .schedule_loom import schedule_loom_calc_model, loom_plans_view
from ..config import logger
router = APIRouter()

@router.post("/plan", response_model=LoomPlansOut)
async def calc_loom_plan(plan_in: DataLoomIn, response: Response):
    logger.debug("start calc plan")
    plan_out = schedule_loom_calc_model(plan_in)
    logger.debug("end calc plan")
    if plan_out.error_str != "":
        response.status_code = 500
        logger.error(plan_out.error_str)
    return plan_out


@router.post("/planview", response_model=LoomPlansViewOut)
async def calc_loom_plan(plan_view_in: LoomPlansViewIn, response: Response):
    logger.debug("start view plan")
    plan_view_out = loom_plans_view(plan_view_in)
    logger.debug("end view plan")
    if plan_view_out.error_str != "":
        response.status_code = 500
        logger.error(plan_view_out.error_str)
    return plan_view_out
