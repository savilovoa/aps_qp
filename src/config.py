# This Python file uses the following encoding: utf-8
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from os import getcwd
import logging
from logging.config import fileConfig

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str
    VERSION: str = Field(default="0.0.1.2")
    DEBUG: bool = Field(default=False)
    PORT: int
    SERVER_NAME: str = Field(default="0.0.0.0")
    BASE_DIR: str = Field(default=getcwd())
    LOOM_MAX_TIME: int = Field(default=600)
    CALC_TEST_DATA: bool = Field(default=False)
    SOLVER_ENUMERATE: bool = Field(default=False)
    SOLVER_ENUMERATE_COUNT: int = Field(default=3)


settings = Settings()


fileConfig(settings.BASE_DIR + r'/logging.ini')
logger = logging.getLogger(settings.PROJECT_NAME)
