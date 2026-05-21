"""Save/load trained models so predict still works after server reload."""

import joblib
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "instance"
CACHE_FILE = CACHE_DIR / "trained_models.joblib"


def save_trained_bundle(feature_names, class_names, label_encoders, scaler, models):
    """Persist preprocessing artifacts and trained models to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_names": feature_names,
            "class_names": class_names,
            "label_encoders": label_encoders,
            "scaler": scaler,
            "models": models,
        },
        CACHE_FILE,
    )


def load_trained_bundle():
    """Load saved bundle from disk; return None on failure or missing file."""
    if not CACHE_FILE.exists():
        return None
    try:
        return joblib.load(CACHE_FILE)
    except Exception:
        return None


def clear_trained_bundle():
    """Remove saved bundle so app will retrain or reload fresh models."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
