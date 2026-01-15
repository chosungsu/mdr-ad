import numpy as np
import torch
from torch.utils.data import Dataset
import io


class SensorSequenceDataset(Dataset):
    """
    CSV 시계열(sensor_data_rms2_fixed.csv)을 불러와
    (seq_len, feature_dim) 시퀀스 단위로 잘라주는 Dataset.

    - 첫 번째 컬럼(key)은 제외하고 나머지 수치 컬럼만 사용합니다.
    - 기본적으로 전체 데이터를 채널별 z-score 정규화합니다.
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: int,
        stride: int = 1,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        # CSV 로드 (헤더 1줄, 콤마 구분)
        raw = self._load_csv(csv_path)
        # key 컬럼(0번)을 제외한 나머지 컬럼만 사용
        features = raw[:, 1:].astype(np.float32)

        if normalize:
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + 1e-6
            features = (features - mean) / std
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None

        self.seq_len = seq_len
        self.stride = stride
        self.input_dim = features.shape[1]

        # 시퀀스 인덱스 미리 계산
        self._indices = []
        num_steps = features.shape[0]
        for start in range(0, num_steps - seq_len + 1, stride):
            self._indices.append(start)

        self._features = features

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self._indices[idx]
        end = start + self.seq_len
        seq = self._features[start:end]  # (seq_len, input_dim)
        return torch.from_numpy(seq)

    @staticmethod
    def _load_csv(csv_path: str) -> np.ndarray:
        """
        - 로컬 경로: 그대로 읽음
        - GCS 경로: gs://bucket/path 또는 gcs://bucket/path 지원
        """
        if csv_path.startswith("gs://") or csv_path.startswith("gcs://"):
            scheme = "gs://" if csv_path.startswith("gs://") else "gcs://"
            rest = csv_path[len(scheme) :]
            if "/" not in rest:
                raise ValueError(f"잘못된 GCS 경로 형식: {csv_path}")
            bucket, blob_path = rest.split("/", 1)

            try:
                from google.cloud import storage  # type: ignore
            except Exception as e:
                raise ImportError(
                    "GCS 경로를 사용하려면 google-cloud-storage가 필요합니다. "
                    "예) pip install google-cloud-storage"
                ) from e

            client = storage.Client()
            text = client.bucket(bucket).blob(blob_path).download_as_text(encoding="utf-8")
            return np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1)

        return np.loadtxt(csv_path, delimiter=",", skiprows=1)
