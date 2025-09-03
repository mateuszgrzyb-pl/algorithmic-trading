from pathlib import Path
from datetime import date
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

ENV_FILE_PATH = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    finance_toolkit_key: str
    analysis_start_date: Optional[date]
    analysis_end_date: Optional[date]

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
    )


settings = Settings()
