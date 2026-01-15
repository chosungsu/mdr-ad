"""
FastAPI 엔드포인트 모음 (packages/backend)

핵심 4개(프론트 API_ENDPOINTS에서 직접 호출):
  - GET  /health
  - GET  /logs?limit=100
  - GET  /realtime/data
  - POST /realtime/data
  - GET  /models/list

참고(프로젝트에 포함된 기타):
  - POST /predict, POST /predict/batch
  - WS   /realtime
  - GET  /model/info
  - GET  /gcp/list

실행:
  - python main.py
  - RELOAD=true PORT=8000 python main.py
  - 또는: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from pathlib import Path
import threading
import csv
import json
import sys
import io
import importlib
import tempfile
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gcp_logic import DEFAULT_BUCKET, GCS, MODELS_PREFIX

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


app = FastAPI(
    title="Bistelligence API (packages/backend)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Startup: set GCP credentials (local key file) =====
GCS_BOOT_OK: bool = False
GCS_BOOT_ERROR: Optional[str] = None

# ===== Dataset (local CSV) =====
DATASET_LOADED: bool = False
DATASET_ERROR: Optional[str] = None
DATASET_PATH: str = ""
DATASET_FIELDS: List[str] = []
DATASET_ROWS: List[Dict[str, str]] = []
DATASET_MEAN: Optional[np.ndarray] = None  # (7,)
DATASET_STD: Optional[np.ndarray] = None   # (7,)

# ===== Models (GCS -> local inference) =====
MODELS_BOOT_OK: bool = False
MODELS_BOOT_ERROR: Optional[str] = None
MODEL_MSCVAE: Any = None
MODEL_TCAD: Any = None
MODEL_MDRAD: Any = None
MSC_SEQ_LEN: int = 50
TCAD_SEQ_LEN: int = 50
MDR_SEQ_LEN: int = 50
MDR_ALPHA_CORR: float = 1.0
MDR_BETA_DISC: float = 0.1
MODEL_DEVICE: str = "cpu"
_SEQ_BUF: "deque[np.ndarray]" = deque(maxlen=50)  # normalized row vectors (7,)
_LATEST_SCORES: Dict[str, Any] = {"timestamp": None, "cursor": 0, "scores": {"mdrad": None}}
_MODEL_LOCK = threading.Lock()

# ===== Logs (local CSV cursor) =====
# 프론트가 cursor를 들고 있고, 백엔드는 "cursor 다음 행"을 반환합니다.
# Stop/Run은 프론트에서 상태를 리셋하므로, 백엔드에 별도 stream 상태/삭제 로직이 필요 없습니다.


def _default_key_path() -> str:
    # packages/backend/bis-ai-1bfdb5128839.json
    return str(Path(__file__).resolve().parent / "bis-ai-1bfdb5128839.json")

def _try_download_gcp_keyfile_from_gcs() -> Optional[str]:
    """
    선택 기능: GCS에 업로드된 서비스 계정 키 파일을 내려받아 로컬에 저장하고 경로를 반환합니다.

    전제:
    - 이 다운로드 작업 자체를 수행할 수 있는 인증(ADC/IAM)이 이미 있어야 합니다.
      (예: Cloud Run/VM 서비스 계정에 storage.objects.get 권한 부여)

    설정 방법(둘 중 하나):
    - GCP_KEYFILE_GCS_URI="gs://bia-ai-storage/keyfile/xxx.json"
    - 또는
      GCP_KEYFILE_GCS_BUCKET="bia-ai-storage"
      GCP_KEYFILE_GCS_PATH="keyfile/xxx.json"
    """
    uri = (os.environ.get("GCP_KEYFILE_GCS_URI") or "").strip()
    bucket = (os.environ.get("GCP_KEYFILE_GCS_BUCKET") or "").strip()
    obj = (os.environ.get("GCP_KEYFILE_GCS_PATH") or "").strip()

    if uri:
        if not uri.startswith("gs://"):
            raise ValueError(f"GCP_KEYFILE_GCS_URI must start with gs:// (got: {uri})")
        rest = uri[len("gs://") :]
        if "/" not in rest:
            raise ValueError(f"Invalid GCS URI (expected gs://bucket/path): {uri}")
        bucket, obj = rest.split("/", 1)

    if not bucket or not obj:
        return None

    # google-cloud-storage의 기본 인증(ADC)을 사용해 다운로드 시도
    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:
        raise RuntimeError("google-cloud-storage가 설치되어 있어야 GCS 키파일 다운로드가 가능합니다.") from e

    client = storage.Client()
    blob = client.bucket(bucket).blob(obj)
    if not blob.exists():
        raise FileNotFoundError(f"GCS keyfile not found: gs://{bucket}/{obj}")

    # OS temp에 저장 (레포/컨테이너 이미지에 남기지 않기)
    local_path = os.path.join(tempfile.gettempdir(), "bistelligence_gcp_keyfile.json")
    blob.download_to_filename(local_path)
    return local_path

def _get_gcs_default() -> GCS:
    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCP_KEY_FILE") or _default_key_path()
    return GCS(bucket=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET), credentials=key_path)

def _default_dataset_path() -> str:
    # (deprecated) 로컬 CSV 기본 경로. 기본 동작은 GCS입니다.
    # 필요하면 DATASET_CSV_PATH 환경변수로 로컬 경로를 지정하세요.
    return str(Path(__file__).resolve().parents[1] / "dataset" / "sensor_data_rms2_fixed.csv")


def _default_dataset_gcs_path() -> str:
    # GCS 객체 경로: gs://<bucket>/data/sensor_data_rms2_fixed.csv
    # bucket은 _get_gcs_default()에서 GCS_BUCKET(없으면 DEFAULT_BUCKET)을 사용합니다.
    return "data/sensor_data_rms2_fixed.csv"

def _load_dataset() -> None:
    """
    로컬 CSV를 메모리에 적재합니다.
    - 컬럼 행(헤더) 제외
    - key 컬럼(대소문자 무시) 제외
    - 나머지 7개 컬럼만 사용
    """
    global DATASET_LOADED, DATASET_ERROR, DATASET_PATH, DATASET_FIELDS, DATASET_ROWS

    try:
        local_csv_path = os.environ.get("DATASET_CSV_PATH")
        if local_csv_path:
            DATASET_PATH = local_csv_path
            if not os.path.exists(local_csv_path):
                raise FileNotFoundError(f"dataset csv not found: {local_csv_path}")
            with open(local_csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("csv has no header")

                key_field: Optional[str] = None
                for name in reader.fieldnames:
                    if name.strip().lower() == "key":
                        key_field = name
                        break

                fields = [n for n in reader.fieldnames if n != key_field]
                if len(fields) > 7:
                    fields = fields[:7]

                rows: List[Dict[str, str]] = []
                for r in reader:
                    item: Dict[str, str] = {}
                    for k in fields:
                        item[k] = (r.get(k) or "").strip()
                    rows.append(item)
        else:
            # 기본: GCS에서 CSV를 내려받아 로드
            gcs = _get_gcs_default()
            gcs_path = os.environ.get("DATASET_GCS_PATH") or _default_dataset_gcs_path()
            text = gcs.download_text(gcs_path, encoding="utf-8")
            if text is None:
                raise FileNotFoundError(f"dataset csv not found on gcs: gs://{gcs.bucket_name}/{gcs_path}")

            DATASET_PATH = f"gs://{gcs.bucket_name}/{gcs_path}"
            reader = csv.DictReader(io.StringIO(text))
            if not reader.fieldnames:
                raise ValueError("csv has no header")

            key_field: Optional[str] = None
            for name in reader.fieldnames:
                if name.strip().lower() == "key":
                    key_field = name
                    break

            fields = [n for n in reader.fieldnames if n != key_field]
            if len(fields) > 7:
                fields = fields[:7]
            if len(fields) != 7:
                # 그래도 7개가 아니면 그대로 진행하되 경고성 에러 기록
                pass

            rows: List[Dict[str, str]] = []
            for r in reader:
                item: Dict[str, str] = {}
                for k in fields:
                    item[k] = (r.get(k) or "").strip()
                rows.append(item)

        DATASET_FIELDS = fields
        DATASET_ROWS = rows

        # 데이터셋 정규화 통계(mean/std) 계산 (학습 코드와 동일: z-score)
        try:
            mat = np.array([[float(r.get(k, "nan") or "nan") for k in fields] for r in rows], dtype=np.float32)
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0) + 1e-6
            globals()["DATASET_MEAN"] = mean
            globals()["DATASET_STD"] = std
        except Exception:
            globals()["DATASET_MEAN"] = None
            globals()["DATASET_STD"] = None

        DATASET_LOADED = True
        DATASET_ERROR = None
    except Exception as e:
        DATASET_LOADED = False
        DATASET_ERROR = str(e)
        DATASET_FIELDS = []
        DATASET_ROWS = []
        globals()["DATASET_MEAN"] = None
        globals()["DATASET_STD"] = None


@app.on_event("startup")
async def _startup_gcp_credentials() -> None:
    """
    서버 시작 시 GCP 자격증명 설정:
    - 이미 GOOGLE_APPLICATION_CREDENTIALS가 있으면 그대로 사용
    - 없으면 packages/backend 폴더의 키 파일을 기본으로 설정
    - 그리고 GCS 접근이 되는지 간단히 확인(list_blobs 1개 시도)
    """
    global GCS_BOOT_OK, GCS_BOOT_ERROR

    try:
        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCP_KEY_FILE")
        if not key_path:
            key_path = _default_key_path()

        if not os.path.exists(key_path):
            # 로컬 키 파일이 없으면, (선택) GCS에서 키 파일을 내려받아 사용
            downloaded = _try_download_gcp_keyfile_from_gcs()
            if downloaded and os.path.exists(downloaded):
                key_path = downloaded
            else:
                GCS_BOOT_OK = False
                GCS_BOOT_ERROR = (
                    f"GCP key file not found locally: {key_path}. "
                    "If you want to download it from GCS, set GCP_KEYFILE_GCS_URI "
                    "or (GCP_KEYFILE_GCS_BUCKET + GCP_KEYFILE_GCS_PATH). "
                    "Note: this requires ADC/IAM permission to read that object."
                )
                return

        # 환경변수로 고정 (google-cloud-storage 기본 인증 경로)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

        # 부팅 시 GCS 접근 테스트:
        # list 권한이 없을 수 있으므로, "알려진 객체" 존재 확인(GET) 위주로 테스트합니다.
        gcs = GCS(bucket=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET), credentials=key_path)
        test_object = os.environ.get("GCS_TEST_OBJECT", "data/sensor_data_rms2_fixed.csv")
        try:
            _ = gcs.bucket.blob(test_object).exists()
        except Exception:
            # exists()도 권한/네트워크에 따라 실패할 수 있으므로 무조건 실패로 보지 않음
            pass

        GCS_BOOT_OK = True
        GCS_BOOT_ERROR = None
    except Exception as e:
        GCS_BOOT_OK = False
        GCS_BOOT_ERROR = str(e)

    # ---- load local dataset ----
    _load_dataset()

    # ---- load models from GCS ----
    _startup_models()


# ===== In-memory realtime buffer =====
realtime_data_buffer: List[Dict[str, Any]] = []
realtime_buffer_lock = threading.Lock()
realtime_buffer_max_size = 200


def _row_as_numbers(row: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = float("nan")
    return out


def _mask_row_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    보안 목적: 로그로 저장될 때 컬럼명을 col1..col7 형태로 마스킹합니다.
    - 값은 그대로 유지
    - 순서는 DATASET_FIELDS(7개)를 우선으로 사용
    """
    masked: Dict[str, Any] = {}
    fields = list(DATASET_FIELDS)

    # DATASET_FIELDS가 정상(7개)이면 그 순서를 따름. 아니면 입력 row의 key 순서를 사용.
    keys = fields if len(fields) == 7 else list(row.keys())
    for i, k in enumerate(keys):
        if i >= 7:
            break
        masked[f"col{i+1}"] = row.get(k)
    return masked


def _ensure_model_import_paths() -> None:
    """
    modeling 폴더의 모델 정의를 backend에서 import 가능하게 sys.path에 추가.
    (주의) 동일한 모듈명(model/utils)이 여러 폴더에 존재하므로, 실제 import는
    _import_class_from_dir()에서 sys.modules 정리와 함께 수행합니다.
    """
    root = Path(__file__).resolve().parents[2]  # repo root
    for d in [root / "modeling" / "mscvae", root / "modeling" / "tcad", root / "modeling" / "mdrad"]:
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))


def _import_class_from_dir(dir_path: Path, module_name: str, class_name: str):
    """
    모델 폴더(예: modeling/mdrad)의 파일 기반 모듈을 안전하게 import 합니다.
    - 동일한 module_name(예: model/utils)이 여러 폴더에 공존하므로,
      import 전에 sys.modules를 정리하고 sys.path를 임시로 조정합니다.
    """
    to_clear = [module_name, "utils", "loss", "models"]
    for m in to_clear:
        if m in sys.modules:
            del sys.modules[m]

    sys.path.insert(0, str(dir_path))
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    finally:
        # sys.path에서 해당 dir를 제거(중복 제거까지 고려)
        sys.path = [p for p in sys.path if p != str(dir_path)]


def _startup_models() -> None:
    """
    GCS에 저장된 모델(PT 체크포인트)을 내려받아 로드합니다.
    - mdrad:  bia-ai-storage/models/mdr_model.pt (대시보드 기본)
    (참고) 기존 MSCVAE/TCAD 로드는 필요 시 확장 가능합니다.
    """
    global MODELS_BOOT_OK, MODELS_BOOT_ERROR, MODEL_MDRAD, MDR_SEQ_LEN, MDR_ALPHA_CORR, MDR_BETA_DISC, MODEL_DEVICE, _SEQ_BUF

    if torch is None:
        MODELS_BOOT_OK = False
        MODELS_BOOT_ERROR = "torch is not installed"
        return

    try:
        _ensure_model_import_paths()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_DEVICE = str(device)

        gcs = _get_gcs_default()
        mdr_path = os.environ.get("MDR_MODEL_GCS_PATH", f"{MODELS_PREFIX}/mdr_model.pt")
        mdr_bytes = gcs.download_bytes(mdr_path)
        if not mdr_bytes:
            raise RuntimeError(f"failed to download MDRAD model from GCS: {mdr_path}")

        mdr_ckpt = torch.load(io.BytesIO(mdr_bytes), map_location=device)
        MDR_SEQ_LEN = int(mdr_ckpt.get("seq_len", int(os.environ.get("MODEL_SEQ_LEN", "50"))))
        input_dim = int(mdr_ckpt.get("input_dim", 7))
        d_model = int(mdr_ckpt.get("d_model", 128))
        transformer_layers = int(mdr_ckpt.get("transformer_layers", 3))
        nhead = int(mdr_ckpt.get("nhead", 4))
        MDR_ALPHA_CORR = float(mdr_ckpt.get("alpha_corr", float(os.environ.get("MDR_ALPHA_CORR", "1.0"))))
        MDR_BETA_DISC = float(mdr_ckpt.get("beta_disc", float(os.environ.get("MDR_BETA_DISC", "0.1"))))

        root = Path(__file__).resolve().parents[2]
        mdrad_dir = root / "modeling" / "mdrad"
        MDRAD = _import_class_from_dir(mdrad_dir, "model", "MDRAD")

        mdr_model = MDRAD(
            seq_len=MDR_SEQ_LEN,
            input_dim=input_dim,
            d_model=d_model,
            transformer_layers=transformer_layers,
            nhead=nhead,
        ).to(device)
        mdr_model.load_state_dict(mdr_ckpt["model_state_dict"])
        mdr_model.eval()

        with _MODEL_LOCK:
            MODEL_MDRAD = mdr_model
            # seq_len 변화에 맞춰 버퍼 길이도 재설정
            _SEQ_BUF = deque(maxlen=MDR_SEQ_LEN)

        MODELS_BOOT_OK = True
        MODELS_BOOT_ERROR = None
    except Exception as e:
        MODELS_BOOT_OK = False
        MODELS_BOOT_ERROR = str(e)
        with _MODEL_LOCK:
            MODEL_MDRAD = None


def _normalize_row(v: np.ndarray) -> np.ndarray:
    if DATASET_MEAN is None or DATASET_STD is None:
        return v
    return (v - DATASET_MEAN) / DATASET_STD


def _build_sequence_tensor(required_len: int) -> Optional["torch.Tensor"]:
    if torch is None:
        return None
    # 현재 버퍼를 기반으로 seq_len을 만들고, 부족하면 첫 값으로 padding
    with _MODEL_LOCK:
        buf = list(_SEQ_BUF)
    if not buf:
        return None
    if len(buf) >= required_len:
        seq = np.stack(buf[-required_len:], axis=0)
    else:
        pad = np.repeat(buf[0][None, :], required_len - len(buf), axis=0)
        seq = np.concatenate([pad, np.stack(buf, axis=0)], axis=0)
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)  # (1, seq, dim)
    device = torch.device(MODEL_DEVICE) if torch is not None else torch.device("cpu")
    return x.to(device)


def _mscvae_score(x: "torch.Tensor") -> Optional[float]:
    if torch is None:
        return None
    with _MODEL_LOCK:
        model = MODEL_MSCVAE
    if model is None:
        return None
    with torch.no_grad():
        recon_temporal, _mus, _logvars = model(x)
        # attribute matrix (eval.py와 동일)
        batch, seq, dim = x.shape
        mats = []
        for t in range(seq):
            xt = x[:, t]  # (batch, dim)
            mats.append(xt.unsqueeze(2) * xt.unsqueeze(1))  # (batch, dim, dim)
        M = torch.stack(mats, dim=1).unsqueeze(2)  # (batch, seq, 1, dim, dim)
        recon_err = torch.mean((M - recon_temporal) ** 2, dim=[2, 3, 4])  # (batch, seq)
        score = torch.mean(recon_err, dim=1)  # (batch,)
        return float(score.squeeze().item())


def _tcad_score(x: "torch.Tensor") -> Optional[float]:
    if torch is None:
        return None
    with _MODEL_LOCK:
        model = MODEL_TCAD
    if model is None:
        return None
    with torch.no_grad():
        x_hat, z1, z2 = model(x)
        recon_err = torch.mean((x - x_hat) ** 2, dim=[1, 2])
        rep_diff = torch.mean((z1 - z2) ** 2, dim=[1, 2])
        score = recon_err + rep_diff
        return float(score.squeeze().item())


def _mdrad_score(x: "torch.Tensor") -> Optional[float]:
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
            torch.mean((z1 - z2) ** 2, dim=1)
            + torch.mean((z1 - z3) ** 2, dim=1)
            + torch.mean((z2 - z3) ** 2, dim=1)
        ) / 3.0
        score = recon_x + alpha_corr * recon_m + beta_disc * disc
        return float(score.squeeze().item())


# ===== (Optional) local model dir listing =====
DEFAULT_LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "")


# ===== Schemas =====
class SensorDataPoint(BaseModel):
    Pressure: float = Field(..., description="압력")
    Power1: float = Field(..., description="전력1")
    Power2: float = Field(..., description="전력2")
    Vibration_Peak1: float = Field(..., description="진동 피크1")
    Vibration_RMS1: float = Field(..., description="진동 RMS1")
    Vibration_Peak2: float = Field(..., description="진동 피크2")
    Vibration_RMS2: float = Field(..., description="진동 RMS2")


class BatchSensorData(BaseModel):
    data: List[SensorDataPoint] = Field(..., description="센서 데이터 리스트")


class PredictionResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold_95: float
    threshold_75: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    anomaly_count: int


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_path: Optional[str] = None
    message: Optional[str] = None


class ModelListResponse(BaseModel):
    success: bool
    models: List[str]
    count: int
    local_model_dir: Optional[str] = None
    gcs_bucket: str = DEFAULT_BUCKET
    gcs_prefix: str = MODELS_PREFIX
    gcs_error: Optional[str] = None


class RealtimeDataResponse(BaseModel):
    timestamp: str
    data: Optional[Dict[str, Any]]
    anomaly_score: Optional[float]
    is_anomaly: Optional[bool]
    status: str


class SystemLogsResponse(BaseModel):
    success: bool
    logs: List[Dict[str, Any]]
    count: int
    message: str
    last_id: int = 0
    next_cursor: int = 0
    wrapped: bool = False


class RealtimeScoresResponse(BaseModel):
    timestamp: str
    cursor: int
    scores: Dict[str, Optional[float]]
    status: str
    error: Optional[str] = None


# ===== Helpers =====
def _now() -> str:
    return datetime.now().isoformat()

def _dummy_anomaly_score(x: Dict[str, Any]) -> float:
    # 모델이 없을 때도 UI가 움직이도록, 간단한 더미 스코어 생성
    v = float(x.get("Vibration_RMS2", 0.0))
    base = np.tanh((v - 7.0) / 5.0)
    noise = (np.random.random() - 0.5) * 0.15
    return float(base + noise)


def _get_gcs_from_request(request: Request) -> GCS:
    # 헤더로 서비스계정 JSON을 주는 흐름을 지원
    key_json = request.headers.get("X-GCP-Key-JSON")
    creds = GCS.parse_key_header(key_json)
    return GCS(bucket=os.environ.get("GCS_BUCKET", DEFAULT_BUCKET), credentials=creds)


# ===== Core endpoints =====
@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "message": "Bistelligence API (packages/backend)",
        "status": "running",
        "timestamp": _now(),
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    system: Dict[str, Any] = {}
    if psutil is not None:
        try:
            system = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
            }
        except Exception:
            system = {}
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
            "mdrad_alpha_corr": MDR_ALPHA_CORR,
            "mdrad_beta_disc": MDR_BETA_DISC,
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
    request: Request,
    limit: int = 1,
    cursor: int = 0,
    wrap: bool = False,
) -> SystemLogsResponse:
    """
    로컬 CSV(packages/dataset)에서 cursor 기반으로 다음 행을 반환합니다.
    - cursor: 마지막으로 반환한 행 번호(0..N). 다음 호출에서 cursor+1 행을 반환.
    - wrap: true면 마지막 행 이후 1행부터 다시 시작(기본 false)
    - Stop/Run은 프론트에서 cursor를 0으로 리셋해서 "첫 행부터"를 구현합니다.
    """
    try:
        _ = request
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
        last = max(0, min(int(cursor), total))

        out: List[Dict[str, Any]] = []
        wrapped_flag = False
        current = last

        # 새로 시작(cursor=0) 또는 wrap으로 1행부터 재시작하는 경우, 모델용 시퀀스 버퍼도 초기화
        if int(cursor) == 0:
            with _MODEL_LOCK:
                _SEQ_BUF.clear()

        for _i in range(limit):
            nxt = current + 1
            if nxt > total:
                if wrap:
                    wrapped_flag = True
                    nxt = 1
                else:
                    break

            row_raw = DATASET_ROWS[nxt - 1]
            row_num = _row_as_numbers(row_raw)
            masked = _mask_row_keys(row_num)
            ts = _now()

            # wrap으로 1행부터 다시 시작하는 순간에는, 로그/차트가 "초기화 후 다시 쌓이기"가 자연스럽게 되도록 버퍼 리셋
            if wrapped_flag and nxt == 1:
                with _MODEL_LOCK:
                    _SEQ_BUF.clear()

            # 모델 점수 계산을 위해 정규화된 row를 시퀀스 버퍼에 push
            try:
                vec = np.array([row_num.get(k, float("nan")) for k in DATASET_FIELDS], dtype=np.float32)
                vec = _normalize_row(vec)
                with _MODEL_LOCK:
                    _SEQ_BUF.append(vec)
            except Exception:
                pass

            # 최신 점수 갱신 (현재 /logs 호출이 "1초마다 한 행"이므로 여기서 갱신하면 dashboard와 동기화됨)
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
                # 점수 계산 실패 시에도 로그는 계속
                pass

            out.append(
                {
                    "id": nxt,
                    "timestamp": ts,
                    "level": "info",
                    "message": json.dumps({"ts": ts, "data": masked}, ensure_ascii=False),
                    "source": "dataset",
                }
            )
            current = nxt

        return SystemLogsResponse(
            success=True,
            logs=out,
            count=len(out),
            message="logs fetched",
            last_id=current,
            next_cursor=current,
            wrapped=wrapped_flag,
        )
    except Exception as e:
        return SystemLogsResponse(
            success=False,
            logs=[],
            count=0,
            message=str(e),
            last_id=int(cursor),
            next_cursor=int(cursor),
            wrapped=False,
        )


@app.get("/realtime/scores", response_model=RealtimeScoresResponse)
async def realtime_scores() -> RealtimeScoresResponse:
    """
    최신 anomaly score를 반환합니다.
    - 점수는 /logs 호출(=1초마다 한 행)에서 갱신되므로, dashboard는 이 엔드포인트만 폴링하면 됩니다.
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
        scores={
            "mdrad": scores.get("mdrad"),
        },
        status="success",
        error=None,
    )


@app.post("/realtime/data")
async def realtime_post(body: SensorDataPoint) -> Dict[str, Any]:
    item = body.model_dump()
    item["timestamp"] = _now()
    with realtime_buffer_lock:
        realtime_data_buffer.append(item)
        if len(realtime_data_buffer) > realtime_buffer_max_size:
            realtime_data_buffer[:] = realtime_data_buffer[-realtime_buffer_max_size :]
    return {"success": True}


@app.get("/models/list", response_model=ModelListResponse)
async def models_list(request: Request) -> ModelListResponse:
    local_files: List[str] = []
    local_dir = DEFAULT_LOCAL_MODEL_DIR or None
    if local_dir and os.path.isdir(local_dir):
        for f in os.listdir(local_dir):
            if f.lower().endswith((".pkl", ".pt", ".pth", ".joblib")):
                local_files.append(f)

    gcs_files: List[str] = []
    gcs_error: Optional[str] = None
    try:
        gcs = _get_gcs_from_request(request)
        gcs_files = gcs.list_models(prefix=MODELS_PREFIX)
    except Exception as e:
        gcs_files = []
        gcs_error = str(e)

    # 보기 좋게 합치기 (중복 제거)
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


# ===== Reference endpoints =====
@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(model_loaded=False, model_path=None, message="model loading not wired in packages/backend")


@app.post("/predict", response_model=PredictionResponse)
async def predict(point: SensorDataPoint, threshold_95: float = 0.75, threshold_75: float = 0.45) -> PredictionResponse:
    data = point.model_dump()
    score = _dummy_anomaly_score(data)
    return PredictionResponse(
        anomaly_score=score,
        is_anomaly=score >= threshold_95,
        threshold_95=threshold_95,
        threshold_75=threshold_75,
        timestamp=_now(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchSensorData, threshold_95: float = 0.75, threshold_75: float = 0.45) -> BatchPredictionResponse:
    preds: List[PredictionResponse] = []
    anomaly_count = 0
    for p in batch.data:
        score = _dummy_anomaly_score(p.model_dump())
        is_anom = score >= threshold_95
        if is_anom:
            anomaly_count += 1
        preds.append(
            PredictionResponse(
                anomaly_score=score,
                is_anomaly=is_anom,
                threshold_95=threshold_95,
                threshold_75=threshold_75,
                timestamp=_now(),
            )
        )
    return BatchPredictionResponse(predictions=preds, total_count=len(preds), anomaly_count=anomaly_count)


@app.websocket("/realtime")
async def realtime_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_json()
            # payload를 SensorDataPoint 형태로 맞추려 시도
            score = _dummy_anomaly_score(payload if isinstance(payload, dict) else {})
            await ws.send_json(
                {
                    "timestamp": _now(),
                    "anomaly_score": score,
                    "is_anomaly": score >= 0.75,
                    "status": "success",
                }
            )
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/gcp/list")
async def gcp_list(request: Request, prefix: str = MODELS_PREFIX) -> Dict[str, Any]:
    try:
        gcs = _get_gcs_from_request(request)
        models = gcs.list_models(prefix=prefix)
        return {"success": True, "bucket": gcs.bucket_name, "prefix": prefix, "models": models, "count": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() in ("1", "true", "yes", "y", "on")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)

