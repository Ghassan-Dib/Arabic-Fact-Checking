from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    fact_check_tools_url: str
    api_key: str
    claude_model: str = "claude-sonnet-4-20250514"
    default_language: str = "ar"
    default_max_age_days: int = 365
    log_level: str = "INFO"
    data_dir: Path = Path("data")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def evidence_dir(self) -> Path:
        return self.data_dir / "evidence"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
