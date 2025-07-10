from idlelib.query import Query

from fastapi import APIRouter, Query
from .model_loom import DataLoomIn, LoomPlansOut, LoomPlansViewIn, LoomPlansViewOut, LoomPlansViewByIdIn
from starlette.responses import Response, HTMLResponse
from .schedule_loom import schedule_loom_calc_model, loom_plans_view
from ..config import logger, settings
import traceback as tr

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
async def calc_loom_plan_view(plan_view_in: LoomPlansViewIn, response: Response):
    logger.debug("start view plan")
    plan_view_out = loom_plans_view(plan_view_in)
    logger.debug("end view plan")
    if plan_view_out.error_str != "":
        response.status_code = 500
        logger.error(plan_view_out.error_str)
    return plan_view_out

@router.get("/plan_view_by_id")
async def calc_loom_plan_view_by_id(id: str = Query(None, max_length=50)):
    logger.debug("start view plan by id")
    try:
        f_name = settings.BASE_DIR + f"/data/{id}.html"

        with open(f_name, "r") as f:
            result = f.read()

        logger.debug("end view plan by id")

    except Exception as e:
        error = tr.TracebackException(exc_type=type(e), exc_traceback=e.__traceback__, exc_value=e).stack[-1]
        error_str = '{} in file {} in {} row:{} '.format(e, error.filename, error.lineno, error.line)
        logger.error(error_str)
        result = error_str
    return HTMLResponse(result)
