"""Central configuration. Every value can be overridden with an environment variable."""
import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env(name, default, cast=str):
    return lambda: cast(os.getenv(name, default))


@dataclass
class Settings:
    # --- Pretrained models (downloaded from Hugging Face on first run) ---
    # FashionCLIP: CLIP fine-tuned on ~800k fashion products (bags, shoes, apparel)
    image_model: str = field(default_factory=_env("IMAGE_MODEL", "patrickjohncyh/fashion-clip"))
    # Multilingual-friendly small sentence encoder for titles/descriptions
    text_model: str = field(default_factory=_env("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    device: str = field(default_factory=_env("DEVICE", "auto"))  # auto | cpu | cuda | mps

    data_dir: str = field(default_factory=_env("DATA_DIR", "data"))
    categories: tuple = field(
        default_factory=lambda: tuple(c.strip() for c in os.getenv("CATEGORIES", "bag,shoe,clothing").split(","))
    )

    # --- Search ---
    top_k: int = field(default_factory=_env("TOP_K", 20, int))
    max_matches: int = field(default_factory=_env("MAX_MATCHES", 5, int))
    compare_same_vendor: bool = field(
        default_factory=lambda: os.getenv("COMPARE_SAME_VENDOR", "false").lower() == "true"
    )

    # --- Scoring weights (normalised automatically) ---
    w_image: float = field(default_factory=_env("W_IMAGE", 0.60, float))
    w_text: float = field(default_factory=_env("W_TEXT", 0.30, float))
    w_price: float = field(default_factory=_env("W_PRICE", 0.10, float))

    # --- Risk thresholds (tune with scripts/calibrate.py once you have reviewed alerts) ---
    high_risk: float = field(default_factory=_env("HIGH_RISK", 0.90, float))
    medium_risk: float = field(default_factory=_env("MEDIUM_RISK", 0.82, float))

    # Perceptual hash: distance <= this means the same photo (re-saved, resized, lightly edited)
    phash_max_distance: int = field(default_factory=_env("PHASH_MAX_DISTANCE", 6, int))
    phash_floor_score: float = field(default_factory=_env("PHASH_FLOOR_SCORE", 0.97, float))


@lru_cache
def get_settings() -> Settings:
    return Settings()
