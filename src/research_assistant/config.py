from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    base_url: str = Field(validation_alias="OPENROUTER_BASE_URL")
    api_key: str = Field(validation_alias="OPENROUTER_API_KEY")
    model_id: str
    max_steps: int
    temperature: float
    timeout: int
