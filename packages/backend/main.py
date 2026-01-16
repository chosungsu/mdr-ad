"""
Bistelligence API Backend

Core Endpoints:
  - GET  /api/health         - Server health check
  - GET  /api/logs           - System logs from CSV dataset
  - GET  /api/realtime/scores - Latest anomaly scores
  - GET  /api/models/list    - List available models

API Documentation:
  - Swagger UI: /docs
  - ReDoc: /redoc

Note: All endpoints are prefixed with /api due to root_path="/api"
"""

import os
import json
import csv
import io
import sys
import tempfile
import threading
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from gcp_logic import DEFAULT_BUCKET, GCS, MODELS_PREFIX

# Load environment variables from .env file
load_dotenv()

try:
    import torch
except Exception:
    torch = None

try:
    import psutil
except Exception:
    psutil = None


# ===== FastAPI App =====
# Use root_path="/api" only in production (behind Nginx proxy)
# Local development: no root_path (direct access to localhost:8000)
USE_ROOT_PATH = os.environ.get("USE_API_PREFIX", "false").lower() in ("1", "true", "yes")

app = FastAPI(
    title="Bistelligence API",
    version="1.0.0",
    root_path="/api" if USE_ROOT_PATH else "",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Global State =====
# GCP Credentials
GCS_BOOT_OK: bool = False
GCS_BOOT_ERROR: Optional[str] = None

# Dataset (CSV from GCS)
DATASET_LOADED: bool = False
DATASET_ERROR: Optional[str] = None
DATASET_PATH: str = ""
DATASET_FIELDS: List[str] = []
DATASET_ROWS: List[Dict[str, str]] = []
DATASET_MEAN: Optional[np.ndarray] = None
DATASET_STD: Optional[np.ndarray] = None

# Model (MDRAD from GCS)
MODELS_BOOT_OK: bool = False
MODELS_BOOT_ERROR: Optional[str] = None
MODEL_MDRAD: Any = None
MODEL_DEVICE: str = "cpu"
MDR_SEQ_LEN: int = 50
MDR_ALPHA_CORR: float = 1.0
MDR_BETA_DISC: float = 0.1

# Sequence buffer for anomaly detection
_SEQ_BUF: "deque[np.ndarray]" = deque(maxlen=50)
_LATEST_SCORES: Dict[str, Any] = {"timestamp": None, "cursor": 0, "scores": {"mdrad": None}}
_MODEL_LOCK = threading.Lock()


# ===== Pydantic Models =====
class SystemLogsResponse(BaseModel):
    """Response model for system logs endpoint (includes realtime scores)"""
    success: bool
    logs: List[Dict[str, Any]]
    count: int
    message: str
    last_id: int = 0
    next_cursor: int = 0
    wrapped: bool = False
    # Realtime scores (integrated for dashboard)
    scores: Dict[str, Optional[float]] = {}
    scores_timestamp: Optional[str] = None
    scores_status: str = "success"


class RealtimeScoresResponse(BaseModel):
    """Response model for realtime scores endpoint"""
    timestamp: str
    cursor: int
    scores: Dict[str, Optional[float]]
    status: str
    error: Optional[str] = None


class ModelListResponse(BaseModel):
    """Response model for models list endpoint"""
    success: bool
    models: List[str]
    count: int
    local_model_dir: Optional[str] = None
    gcs_bucket: str = DEFAULT_BUCKET
    gcs_prefix: str = MODELS_PREFIX
    gcs_error: Optional[str] = None


# ===== Startup: Initialize GCP Credentials =====
@app.on_event("startup")
async def _startup_gcp_credentials() -> None:
    """
    Initialize GCP credentials on server startup.
    
    Reads GCP_* environment variables from .env file and reconstructs
    service account JSON for authentication.
    
    Required environment variables:
    - GCP_PROJECT_ID
    - GCP_PRIVATE_KEY
    - GCP_PRIVATE_KEY_ID
    - GCP_CLIENT_EMAIL
    - GCP_CLIENT_ID
    """
    global GCS_BOOT_OK, GCS_BOOT_ERROR

    try:
        # Check required environment variables
        required_vars = ["GCP_PROJECT_ID", "GCP_PRIVATE_KEY", "GCP_CLIENT_EMAIL"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            GCS_BOOT_OK = False
            GCS_BOOT_ERROR = f"Missing required environment variables: {', '.join(missing_vars)}"
            print(f"[GCP] ERROR: {GCS_BOOT_ERROR}")
            print(f"[GCP] Please set these variables in .env file")
            return

        # Reconstruct service account JSON from environment variables
        service_account_info = {
            "type": os.environ.get("GCP_TYPE", "service_account"),
            "project_id": os.environ.get("GCP_PROJECT_ID"),
            "private_key_id": os.environ.get("GCP_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("GCP_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.environ.get("GCP_CLIENT_EMAIL"),
            "client_id": os.environ.get("GCP_CLIENT_ID"),
            "auth_uri": os.environ.get("GCP_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": os.environ.get("GCP_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": os.environ.get("GCP_AUTH_PROVIDER_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
            "client_x509_cert_url": os.environ.get("GCP_CLIENT_CERT_URL"),
            "universe_domain": os.environ.get("GCP_UNIVERSE_DOMAIN", "googleapis.com"),
        }
        
        # Write JSON to temporary file
        key_path = os.path.join(tempfile.gettempdir(), "bistelligence_gcp_keyfile.json")
        with open(key_path, 'w') as f:
            json.dump(service_account_info, f)
        
        # Set GOOGLE_APPLICATION_CREDENTIALS for google-cloud-storage
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        print(f"[GCP] Credentials loaded from environment variables (.env file)")
        print(f"[GCP] Project: {os.environ.get('GCP_PROJECT_ID')}")
        print(f"[GCP] Bucket: {os.environ.get('GCS_BUCKET', DEFAULT_BUCKET)}")

        # Test GCS access
        gcs = GCS(bucket=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET), credentials=key_path)
        test_object = os.environ.get("GCS_TEST_OBJECT", "data/sensor_data_rms2_fixed.csv")
        try:
            _ = gcs.bucket.blob(test_object).exists()
            print(f"[GCP] Successfully connected to GCS")
        except Exception as e:
            print(f"[GCP] Warning: Could not verify GCS access: {e}")

        GCS_BOOT_OK = True
        GCS_BOOT_ERROR = None
    except Exception as e:
        GCS_BOOT_OK = False
        GCS_BOOT_ERROR = str(e)
        print(f"[GCP] ERROR: {e}")

    # Load dataset and models
    _load_dataset()
    _startup_models()


# ===== Dataset Loading =====
def _load_dataset() -> None:
    """Load CSV dataset from GCS into memory"""
    global DATASET_LOADED, DATASET_ERROR, DATASET_PATH, DATASET_FIELDS, DATASET_ROWS, DATASET_MEAN, DATASET_STD

    try:
        gcs = GCS(bucket=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET), credentials=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        gcs_path = os.environ.get("DATASET_GCS_PATH", "data/sensor_data_rms2_fixed.csv")
        text = gcs.download_text(gcs_path, encoding="utf-8")
        
        if text is None:
            raise FileNotFoundError(f"Dataset not found: gs://{gcs.bucket_name}/{gcs_path}")

        DATASET_PATH = f"gs://{gcs.bucket_name}/{gcs_path}"
        reader = csv.DictReader(io.StringIO(text))
        
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        # Exclude 'key' field
        key_field = next((n for n in reader.fieldnames if n.strip().lower() == "key"), None)
        fields = [n for n in reader.fieldnames if n != key_field][:7]

        rows = []
        for r in reader:
            item = {k: (r.get(k) or "").strip() for k in fields}
            rows.append(item)

        DATASET_FIELDS = fields
        DATASET_ROWS = rows

        # Calculate normalization statistics (z-score)
        mat = np.array([[float(r.get(k, "nan") or "nan") for k in fields] for r in rows], dtype=np.float32)
        DATASET_MEAN = np.nanmean(mat, axis=0)
        DATASET_STD = np.nanstd(mat, axis=0) + 1e-6

        DATASET_LOADED = True
        DATASET_ERROR = None
        print(f"[Dataset] Loaded {len(rows)} rows from {DATASET_PATH}")
    except Exception as e:
        DATASET_LOADED = False
        DATASET_ERROR = str(e)
        print(f"[Dataset] ERROR: {e}")


# ===== Model Loading =====
def _startup_models() -> None:
    """Load MDRAD model from GCS"""
    global MODELS_BOOT_OK, MODELS_BOOT_ERROR, MODEL_MDRAD, MODEL_DEVICE, MDR_SEQ_LEN, MDR_ALPHA_CORR, MDR_BETA_DISC, _SEQ_BUF

    if torch is None:
        MODELS_BOOT_OK = False
        MODELS_BOOT_ERROR = "torch is not installed"
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_DEVICE = str(device)

        gcs = GCS(bucket=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET), credentials=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        mdr_path = os.environ.get("MDR_MODEL_GCS_PATH", f"{MODELS_PREFIX}/mdr_model.pt")
        mdr_bytes = gcs.download_bytes(mdr_path)
        
        if not mdr_bytes:
            raise RuntimeError(f"Failed to download MDRAD model: {mdr_path}")

        mdr_ckpt = torch.load(io.BytesIO(mdr_bytes), map_location=device)
        MDR_SEQ_LEN = int(mdr_ckpt.get("seq_len", 50))
        MDR_ALPHA_CORR = float(mdr_ckpt.get("alpha_corr", 1.0))
        MDR_BETA_DISC = float(mdr_ckpt.get("beta_disc", 0.1))

        # Import MDRAD model class
        root = Path(__file__).resolve().parents[2]
        mdrad_dir = root / "modeling" / "mdrad"
        sys.path.insert(0, str(mdrad_dir))
        
        from model import MDRAD
        
        mdr_model = MDRAD(
            seq_len=MDR_SEQ_LEN,
            input_dim=int(mdr_ckpt.get("input_dim", 7)),
            d_model=int(mdr_ckpt.get("d_model", 128)),
            transformer_layers=int(mdr_ckpt.get("transformer_layers", 3)),
            nhead=int(mdr_ckpt.get("nhead", 4)),
        ).to(device)
        mdr_model.load_state_dict(mdr_ckpt["model_state_dict"])
        mdr_model.eval()

        with _MODEL_LOCK:
            MODEL_MDRAD = mdr_model
            _SEQ_BUF = deque(maxlen=MDR_SEQ_LEN)

        MODELS_BOOT_OK = True
        MODELS_BOOT_ERROR = None
        print(f"[Model] MDRAD loaded successfully (device={device})")
    except Exception as e:
        MODELS_BOOT_OK = False
        MODELS_BOOT_ERROR = str(e)
        print(f"[Model] ERROR: {e}")


# ===== Helper Functions =====
def _now() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def _row_as_numbers(row: Dict[str, str]) -> Dict[str, float]:
    """Convert string values to floats"""
    return {k: float(v) if v else float("nan") for k, v in row.items()}


def _mask_row_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """Mask column names as col1..col7 for security"""
    masked = {}
    keys = DATASET_FIELDS if len(DATASET_FIELDS) == 7 else list(row.keys())
    for i, k in enumerate(keys[:7]):
        masked[f"col{i+1}"] = row.get(k)
    return masked


def _normalize_row(v: np.ndarray) -> np.ndarray:
    """Normalize row using dataset mean/std (z-score)"""
    if DATASET_MEAN is None or DATASET_STD is None:
        return v
    return (v - DATASET_MEAN) / DATASET_STD


def _build_sequence_tensor(required_len: int) -> Optional["torch.Tensor"]:
    """Build sequence tensor from buffer with padding if needed"""
    if torch is None:
        return None
    
    with _MODEL_LOCK:
        buf = list(_SEQ_BUF)
    
    if not buf:
        return None
    
    if len(buf) >= required_len:
        seq = np.stack(buf[-required_len:], axis=0)
    else:
        pad = np.repeat(buf[0][None, :], required_len - len(buf), axis=0)
        seq = np.concatenate([pad, np.stack(buf, axis=0)], axis=0)
    
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)
    device = torch.device(MODEL_DEVICE)
    return x.to(device)


def _mdrad_score(x: "torch.Tensor") -> Optional[float]:
    """Calculate MDRAD anomaly score"""
    if torch is None:
        return None
    
    with _MODEL_LOCK:
        model = MODEL_MDRAD
        alpha_corr = float(MDR_ALPHA_CORR)
        beta_disc = float(MDR_BETA_DISC)
    
    if model is None:
        return None
    
    with torch.no_grad():
        x_hat, m_hat, m_avg, z1, z2, z3, _z, _w = model(x)
        recon_x = torch.mean((x_hat - x) ** 2, dim=(1, 2))
        recon_m = torch.mean((m_hat - m_avg) ** 2, dim=(1, 2))
        disc = (
            torch.mean((z1 - z2) ** 2, dim=1) +
            torch.mean((z1 - z3) ** 2, dim=1) +
            torch.mean((z2 - z3) ** 2, dim=1)
        ) / 3.0
        score = recon_x + alpha_corr * recon_m + beta_disc * disc
        return float(score.squeeze().item())


# ===== API Endpoints =====
@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint - API status"""
    return {
        "message": "Bistelligence API",
        "status": "running",
        "timestamp": _now(),
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint with system, GCS, model, and dataset status"""
    system = {}
    if psutil is not None:
        try:
            system = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
            }
        except Exception:
            pass

    return {
        "status": "healthy",
        "model_loaded": bool(MODELS_BOOT_OK and MODEL_MDRAD is not None),
        "timestamp": _now(),
        "system": system or None,
        "gcs": {
            "boot_ok": GCS_BOOT_OK,
            "boot_error": GCS_BOOT_ERROR,
            "bucket": os.environ.get("GCS_BUCKET", DEFAULT_BUCKET),
        },
        "models": {
            "boot_ok": MODELS_BOOT_OK,
            "boot_error": MODELS_BOOT_ERROR,
            "device": MODEL_DEVICE,
            "mdrad_seq_len": MDR_SEQ_LEN,
        },
        "dataset": {
            "loaded": DATASET_LOADED,
            "error": DATASET_ERROR,
            "path": DATASET_PATH,
            "rows": len(DATASET_ROWS),
            "fields": DATASET_FIELDS,
        },
    }


@app.get("/logs", response_model=SystemLogsResponse)
async def logs(
    limit: int = 1,
    cursor: int = 0,
    wrap: bool = False,
) -> SystemLogsResponse:
    """
    Get system logs from CSV dataset with cursor-based pagination.
    
    Args:
        limit: Number of rows to return (1-200)
        cursor: Last returned row number (0-N)
        wrap: If true, restart from row 1 after reaching end
    
    Returns:
        System logs response with data and metadata
    """
    try:
        limit = max(1, min(int(limit), 200))

        if not DATASET_LOADED or not DATASET_ROWS:
            return SystemLogsResponse(
                success=False,
                logs=[],
                count=0,
                message=DATASET_ERROR or "dataset not loaded",
                last_id=int(cursor),
                next_cursor=int(cursor),
                wrapped=False,
            )

        total = len(DATASET_ROWS)
        current = max(0, min(int(cursor), total))
        out = []
        wrapped_flag = False

        # Reset sequence buffer on restart
        if int(cursor) == 0:
            with _MODEL_LOCK:
                _SEQ_BUF.clear()

        for _ in range(limit):
            nxt = current + 1
            if nxt > total:
                if wrap:
                    wrapped_flag = True
                    nxt = 1
                    # Reset buffer when wrapping back to start
                    with _MODEL_LOCK:
                        _SEQ_BUF.clear()
                else:
                    break

            # Skip if we've already processed this row (avoid infinite loop)
            if nxt == cursor and _ > 0:
                break

            row_raw = DATASET_ROWS[nxt - 1]
            row_num = _row_as_numbers(row_raw)
            masked = _mask_row_keys(row_num)
            ts = _now()

            # Reset buffer on wrap
            if wrapped_flag and nxt == 1:
                with _MODEL_LOCK:
                    _SEQ_BUF.clear()

            # Add normalized row to sequence buffer
            try:
                vec = np.array([row_num.get(k, float("nan")) for k in DATASET_FIELDS], dtype=np.float32)
                vec = _normalize_row(vec)
                with _MODEL_LOCK:
                    _SEQ_BUF.append(vec)
            except Exception:
                pass

            # Calculate anomaly score
            try:
                mdr_x = _build_sequence_tensor(MDR_SEQ_LEN)
                mdrad_score = _mdrad_score(mdr_x) if mdr_x is not None else None
                with _MODEL_LOCK:
                    globals()["_LATEST_SCORES"] = {
                        "timestamp": ts,
                        "cursor": int(nxt),
                        "scores": {"mdrad": mdrad_score},
                    }
            except Exception:
                pass

            out.append({
                "id": nxt,
                "timestamp": ts,
                "level": "info",
                "message": json.dumps({"ts": ts, "data": masked}, ensure_ascii=False),
                "source": "dataset",
            })
            current = nxt

        # Get latest scores for dashboard integration
        with _MODEL_LOCK:
            latest_scores = dict(_LATEST_SCORES.get("scores") or {})
            scores_ts = _LATEST_SCORES.get("timestamp")
        
        scores_status = "success" if MODELS_BOOT_OK else "error"

        return SystemLogsResponse(
            success=True,
            logs=out,
            count=len(out),
            message="logs fetched",
            last_id=current,
            next_cursor=current,
            wrapped=wrapped_flag,
            scores=latest_scores,
            scores_timestamp=scores_ts,
            scores_status=scores_status,
        )
    except Exception as e:
        # Get latest scores even on error
        with _MODEL_LOCK:
            latest_scores = dict(_LATEST_SCORES.get("scores") or {})
            scores_ts = _LATEST_SCORES.get("timestamp")
        
        return SystemLogsResponse(
            success=False,
            logs=[],
            count=0,
            message=str(e),
            last_id=int(cursor),
            next_cursor=int(cursor),
            wrapped=False,
            scores=latest_scores,
            scores_timestamp=scores_ts,
            scores_status="error",
        )


@app.get("/realtime/scores", response_model=RealtimeScoresResponse)
async def realtime_scores() -> RealtimeScoresResponse:
    """
    Get latest anomaly scores.
    
    Scores are updated by /logs endpoint (called every second).
    Dashboard polls this endpoint to get the latest scores.
    """
    with _MODEL_LOCK:
        ts = _LATEST_SCORES.get("timestamp")
        cursor = int(_LATEST_SCORES.get("cursor") or 0)
        scores = dict(_LATEST_SCORES.get("scores") or {})

    if not MODELS_BOOT_OK:
        return RealtimeScoresResponse(
            timestamp=ts or _now(),
            cursor=cursor,
            scores={"mdrad": None},
            status="error",
            error=MODELS_BOOT_ERROR or "models not loaded",
        )

    return RealtimeScoresResponse(
        timestamp=ts or _now(),
        cursor=cursor,
        scores={"mdrad": scores.get("mdrad")},
        status="success",
        error=None,
    )


@app.get("/models/list", response_model=ModelListResponse)
async def models_list(request: Request) -> ModelListResponse:
    """
    List available models from local directory and GCS.
    
    Returns:
        List of model files with metadata
    """
    local_files = []
    local_dir = os.environ.get("LOCAL_MODEL_DIR")
    if local_dir and os.path.isdir(local_dir):
        for f in os.listdir(local_dir):
            if f.lower().endswith((".pkl", ".pt", ".pth", ".joblib")):
                local_files.append(f)

    gcs_files = []
    gcs_error = None
    try:
        gcs = GCS(bucket=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET), credentials=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        gcs_files = gcs.list_models(prefix=MODELS_PREFIX)
    except Exception as e:
        gcs_error = str(e)

    merged = sorted(set(local_files + [f"[GCS] {m}" for m in gcs_files]))
    return ModelListResponse(
        success=True,
        models=merged,
        count=len(merged),
        local_model_dir=local_dir,
        gcs_bucket=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET),
        gcs_prefix=MODELS_PREFIX,
        gcs_error=gcs_error,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() in ("1", "true", "yes", "y", "on")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
