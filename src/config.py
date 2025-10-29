import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application configuration"""
    
    # Project Settings
    PROJECT_NAME: str = "AI Health Information Assistant"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Model Settings
    INTENT_MODEL_NAME: str = "distilbert-base-uncased"
    QA_MODEL_NAME: str = "dmis-lab/biobert-base-cased-v1.2"
    MAX_LENGTH: int = 512
    DEVICE: str = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"
    
    # API Keys (load from environment)
    OPENFDA_API_KEY: Optional[str] = None
    WHO_API_KEY: Optional[str] = None
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Intent Categories
    INTENT_CATEGORIES: list = [
        "medicine_info",
        "nutrition_advice",
        "general_faq",
        "symptoms_diagnosis",
        "treatment_procedure"
    ]
    
    # Cache Settings
    USE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def create_directories(self):
        """Create necessary directories"""
        for dir_path in [self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.MODELS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.create_directories()