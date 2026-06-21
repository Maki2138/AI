import io
import logging
import os
import math
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .deploy import load_bundle, predict_feature_frame
from .baseline import add_personal_baseline_features

# --- CONFIG ---
MODEL_PATH       = "outputs/models/binary_best_model.joblib"
STATIC_DIR       = Path(__file__).parent / "static"
BASELINE_MINUTES = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stress Detection API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Serve static files ---
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static/")
    return FileResponse(str(index))

# --- Model load ---
MODEL_BUNDLE = None

@app.on_event("startup")
def startup():
    global MODEL_BUNDLE
    try:
        MODEL_BUNDLE = load_bundle(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Cannot load model: {e}")


def _needs_baseline(df: pd.DataFrame) -> bool:
    return not any(col.endswith("_delta_base") for col in df.columns)


def _sanitize(val):
    """Đổi NaN / Inf thành None để json.dumps không bị lỗi."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


# --- API ---
@app.post("/api/v1/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    content = await file.read()
    try:
        features_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read CSV: {e}")

    if features_df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    logger.info(f"File received: {len(features_df)} rows, {len(features_df.columns)} cols")

    # Preprocessing: tính baseline nếu CSV chưa có
    if _needs_baseline(features_df):
        logger.info("Running add_personal_baseline_features()...")
        try:
            features_df = add_personal_baseline_features(
                features_df, baseline_minutes=BASELINE_MINUTES
            )
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Baseline feature error: {e}")

    # Dự đoán
    try:
        results_df = predict_feature_frame(MODEL_BUNDLE, features_df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Prediction error: {e}")

    # Đồng nhất tên cột
    if "decision_adjusted_label" in results_df.columns:
        results_df["adjusted_label"] = results_df["decision_adjusted_label"]
    elif "adjusted_label" not in results_df.columns:
        results_df["adjusted_label"] = 0

    for col in ["alert_state", "recommendation"]:
        if col not in results_df.columns:
            results_df[col] = "N/A"
    if "confidence" not in results_df.columns:
        results_df["confidence"] = 0.0

    # Chỉ giữ cột frontend cần
    output_cols = [c for c in [
        "window_start_sec", "window_start_ts",
        "pred_label", "adjusted_label", "confidence",
        "alert_state", "recommendation",
    ] if c in results_df.columns]
    output_cols += [c for c in results_df.columns if c.startswith("proba_")]

    # Sanitize NaN/Inf → None (null trong JSON) để tránh ValueError
    records = results_df[output_cols].to_dict(orient="records")
    records = [
        {k: _sanitize(v) for k, v in row.items()}
        for row in records
    ]

    logger.info(f"Returning {len(records)} rows")
    return records