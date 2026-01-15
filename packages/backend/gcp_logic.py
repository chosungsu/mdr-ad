"""
GCS 로직 (packages/backend)

버킷: bia-ai-storage
prefix:
  - models/...
  - logs/...

프론트에서 설정한 엔드포인트를 통해 백엔드가 호출되며,
여기서는 GCS 접근/목록/텍스트 다운로드 등의 공통 로직을 제공합니다.
"""

from __future__ import annotations

import os
import base64
import json
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from google.cloud import storage
from google.oauth2 import service_account


DEFAULT_BUCKET = "bia-ai-storage"
MODELS_PREFIX = "models"
LOGS_PREFIX = "logs"


def _maybe_decode_key_json(value: str) -> str:
    """
    프론트/프록시에서 X-GCP-Key-JSON 형태로 전달되는 값을 정규화:
    - base64 인코딩된 JSON 문자열일 수 있음
    - URL 인코딩이 섞일 수 있음
    """
    raw = value.strip()
    if not raw:
        return raw

    # base64 인코딩일 수 있음 (JSON이 아니면 decode 시도)
    if not raw.startswith("{"):
        try:
            decoded = base64.b64decode(raw).decode("utf-8")
            raw = decoded
        except Exception:
            pass

    # URL 인코딩이 섞인 경우
    if "%" in raw:
        try:
            raw = urllib.parse.unquote(raw)
        except Exception:
            pass

    return raw.strip()


def build_storage_client(credentials: Optional[Union[str, Dict[str, Any]]] = None) -> storage.Client:
    """
    credentials:
      - None: GOOGLE_APPLICATION_CREDENTIALS 또는 ADC 사용
      - str:
          - 서비스계정 JSON 문자열
          - 서비스계정 JSON 파일 경로
      - dict: 서비스계정 dict
    """
    if credentials is None:
        return storage.Client()

    if isinstance(credentials, str):
        # 파일 경로면 파일 기반 credentials 생성
        if os.path.exists(credentials):
            creds = service_account.Credentials.from_service_account_file(credentials)
            return storage.Client(credentials=creds, project=creds.project_id)

        # 아니면 JSON 문자열로 처리
        creds_dict = json.loads(credentials)
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        return storage.Client(credentials=creds, project=creds.project_id)

    creds = service_account.Credentials.from_service_account_info(credentials)
    return storage.Client(credentials=creds, project=creds.project_id)


@dataclass
class FileMeta:
    name: str
    full_path: str
    size: Optional[int]
    time_created: Optional[str]
    updated: Optional[str]
    content_type: Optional[str]
    public_url: Optional[str]
    content: Optional[str] = None


class GCS:
    def __init__(
        self,
        bucket: str = DEFAULT_BUCKET,
        credentials: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> None:
        self.bucket_name = bucket
        self.client = build_storage_client(credentials)
        self.bucket = self.client.bucket(bucket)

    @staticmethod
    def parse_key_header(header_value: Optional[str]) -> Optional[str]:
        """
        X-GCP-Key-JSON 헤더에서 JSON 문자열을 뽑아 반환 (없으면 None)
        """
        if not header_value:
            return None
        normalized = _maybe_decode_key_json(header_value)
        if not normalized.startswith("{"):
            return None
        return normalized

    def list_files_with_metadata(self, prefix: str) -> List[FileMeta]:
        blobs = list(self.bucket.list_blobs(prefix=prefix.rstrip("/") + "/"))
        out: List[FileMeta] = []
        for b in blobs:
            if b.name.endswith("/"):
                continue
            out.append(
                FileMeta(
                    name=b.name.split("/")[-1],
                    full_path=b.name,
                    size=getattr(b, "size", None),
                    time_created=b.time_created.isoformat() if getattr(b, "time_created", None) else None,
                    updated=b.updated.isoformat() if getattr(b, "updated", None) else None,
                    content_type=getattr(b, "content_type", None),
                    public_url=f"https://storage.googleapis.com/{self.bucket_name}/{b.name}",
                )
            )
        return out

    def download_text(self, full_path: str, encoding: str = "utf-8") -> Optional[str]:
        try:
            blob = self.bucket.blob(full_path)
            return blob.download_as_text(encoding=encoding)
        except Exception:
            return None

    def download_bytes(self, full_path: str) -> Optional[bytes]:
        try:
            blob = self.bucket.blob(full_path)
            return blob.download_as_bytes()
        except Exception:
            return None

    def upload_text(self, full_path: str, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        blob = self.bucket.blob(full_path)
        blob.upload_from_string(text, content_type=content_type)

    def delete_prefix(self, prefix: str) -> Dict[str, Any]:
        """
        prefix 하위(폴더처럼 쓰는 경로)의 객체를 전부 삭제합니다.
        반환: {"success": bool, "prefix": str, "deleted": int, "errors": [str]}
        """
        p = prefix.strip("/")
        if p and not p.endswith("/"):
            p = p + "/"

        deleted = 0
        errors: List[str] = []
        try:
            for blob in self.bucket.list_blobs(prefix=p):
                try:
                    blob.delete()
                    deleted += 1
                except Exception as e:
                    errors.append(f"{blob.name}: {e}")
        except Exception as e:
            return {"success": False, "prefix": p, "deleted": deleted, "errors": errors + [str(e)]}

        return {"success": len(errors) == 0, "prefix": p, "deleted": deleted, "errors": errors}

    # ===== high-level helpers =====

    def list_models(self, prefix: str = MODELS_PREFIX) -> List[str]:
        files = self.list_files_with_metadata(prefix)
        return [f.full_path.replace(prefix.rstrip("/") + "/", "", 1) for f in files]

    def list_logs(self, key_folder: Optional[str] = None, limit: int = 100) -> List[FileMeta]:
        base = LOGS_PREFIX
        prefix = f"{base}/{key_folder}".strip("/") if key_folder else base
        files = self.list_files_with_metadata(prefix)
        # 최신순 정렬(가능하면 updated 기준)
        def _key(m: FileMeta) -> str:
            return m.updated or m.time_created or ""
        files.sort(key=_key, reverse=True)
        return files[: max(0, limit)]

    def write_log_text(self, full_path: str, lines: List[str]) -> None:
        # full_path는 logs/... 형태를 가정
        blob = self.bucket.blob(full_path)
        blob.upload_from_string("\n".join(lines), content_type="text/plain; charset=utf-8")

    def now_ts(self) -> str:
        return datetime.now().isoformat()

