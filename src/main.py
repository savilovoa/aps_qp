from .config import settings, logger
import uvicorn
from fastapi import FastAPI
from .routers import router
from src.loom.schedule_loom import calc_test_data_from

app = FastAPI()

app.include_router(router, prefix="/api")

def main():
    global app
    if not settings.CALC_TEST_DATA:
        logger.info(f"Start server {settings.SERVER_NAME}")
        uvicorn.run(app, host=settings.SERVER_NAME, port=settings.PORT)
    else:
        calc_test_data_from()


if __name__ == "__main__":
    print(settings.CALC_TEST_DATA)
    main()
