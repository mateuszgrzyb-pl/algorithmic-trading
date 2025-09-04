from datetime import date
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


# --- Nested Configuration Models ---
# Grouping related parameters into sub-models enhances clarity and organization.

class LabelParams(BaseModel):
    """Parameters for the triple-barrier labeling process."""
    profit_targets: List[int] = [1000]
    stop_loses: List[int] = [100]
    max_days: List[int] = [250]
    overwrite: bool = True


class OverlapParams(BaseModel):
    """Parameters for removing overlapping observations."""
    label_time: int = 5


# --- Main Settings Class ---
# This class aggregates all configuration for the application.

class Settings(BaseSettings):
    """
    Main application settings.

    Aggregates all configuration parameters and loads them from environment
    variables or a .env file, providing a single, type-safe source of truth.
    """
    # --- Existing Settings ---
    finance_toolkit_key: str
    analysis_start_date: Optional[date] = None
    analysis_end_date: Optional[date] = None

    # --- New Pipeline Settings ---
    base_path: Path = Path("data")
    target_label_name: str = "label_1000_100_250"

    # --- Nested Configuration Models ---
    label_params: LabelParams = LabelParams()
    overlap_params: OverlapParams = OverlapParams()

    # --- Pydantic Model Configuration ---
    # Specifies how to load settings, including the .env file path
    # and the delimiter for nested environment variables.
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        env_nested_delimiter='__'  # Crucial for overriding nested settings
    )


# A single, global instance of the settings to be imported across the application.
settings = Settings()
