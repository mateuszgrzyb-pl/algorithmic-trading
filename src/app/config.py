from pathlib import Path
from pydantic_settings import BaseSettings

ENV_FILE_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    finance_toolkit_key: str
    analysis_start_date: str
    analysis_end_date: str

    class Config:
        env_file = ENV_FILE_PATH if ENV_FILE_PATH.exists() else None
        env_file_encoding = "utf-8"


settings = Settings()
