import io
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SensorSequenceDataset(Dataset):
    """
    modeling/data/sensor_data_rms2_fixed.csv 기반 시퀀스 Dataset
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
        raw = self._load_csv(csv_path)  # (N, 1+dim)
        features = raw[:, 1:].astype(np.float32)

        self.mean: Optional[np.ndarray]
        self.std: Optional[np.ndarray]
        if normalize:
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True) + 1e-6
            features = (features - mean) / std
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None

        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.input_dim = int(features.shape[1])

        self._indices: list[int] = []
        num_steps = int(features.shape[0])
        for start in range(0, num_steps - self.seq_len + 1, self.stride):
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
        # 최소 통합 버전: 로컬 경로만 지원
        # (필요하면 gs:// 지원을 나중에 다시 붙일 수 있음)
        return np.loadtxt(csv_path, delimiter=",", skiprows=1)

