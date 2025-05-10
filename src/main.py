import datetime
from .config import settings, logger
import uvicorn
from fastapi import FastAPI
#from starlette import status
#from starlette.responses import Response, JSONResponse
from .routers import router

app = FastAPI()

app.include_router(router, prefix="/api")


# @app.post("/calckpv/", response_model=PlanOut)
# async def calc_dye_plan(plan_in: PlanIn, response: Response):
#     plan_out = calc_kpv(plan_in)
#     if plan_out.error != "":
#         response.status_code = 500
#     return plan_out


def main():
    global app
    logger.info(f"Start server {settings.SERVER_NAME}")
    uvicorn.run(app, host=settings.SERVER_NAME, port=settings.PORT)

if __name__ == "__main__":
    main()
