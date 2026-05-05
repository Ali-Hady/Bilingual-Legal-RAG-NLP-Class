from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    MONGO_URI: str
    DB_NAME: str

    ENGLISH_COLLECTION: str = "english_laws"
    ARABIC_COLLECTION: str = "arabic_laws"

    ENGLISH_SEED: Path = BASE_DIR / "seed_data" / "english_laws.json"
    ARABIC_SEED: Path = BASE_DIR / "seed_data" / "arabic_laws.json"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()