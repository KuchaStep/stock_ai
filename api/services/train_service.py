import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_model():

    result = subprocess.run(
        [
            "python",
            "scripts/train_model_v4.py",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "message": result.stderr,
        }

    return {
        "status": "success",
        "message": "Model training completed.",
    }