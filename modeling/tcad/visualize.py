import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SensorSequenceDataset
from eval import anomaly_score, evaluate
from model import TCAD


def _default_data_path() -> str:
    # modeling/tcad 기준: ../data/sensor_data_rms2_fixed.csv -> modeling/data/...
    return "../data/sensor_data_rms2_fixed.csv"


def configure_korean_matplotlib_font() -> None:
    """
    matplotlib에서 한글이 깨지지 않도록 폰트 설정(Windows 우선).
    - Windows: Malgun Gothic
    - Linux: NanumGothic
    - macOS: AppleGothic
    """
    # 음수 기호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False

    try:
        from matplotlib import font_manager as fm

        available = {f.name for f in fm.fontManager.ttflist}
        candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
        for name in candidates:
            if name in available:
                plt.rcParams["font.family"] = name
                return
    except Exception:
        # 폰트 매니저 조회 실패 시, 최소한 Windows 기본값을 시도
        plt.rcParams["font.family"] = "Malgun Gothic"


def contiguous_ranges_from_mask(x: np.ndarray, mask: np.ndarray) -> list[tuple[float, float, float, int]]:
    """
    x축 값과 boolean mask를 받아, True가 연속인 구간을 (start_x, end_x, length, count)로 반환.
    length는 x축 단위로 계산하며, dt는 x의 중앙값 간격을 사용합니다.
    """
    if len(x) == 0:
        return []
    if len(x) != len(mask):
        raise ValueError("x와 mask 길이가 다릅니다.")

    if len(x) >= 2:
        diffs = np.diff(x)
        dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
    else:
        dt = 1.0

    ranges: list[tuple[float, float, float, int]] = []
    in_run = False
    start_i = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            in_run = True
            start_i = i
        elif (not v) and in_run:
            end_i = i - 1
            start_x = float(x[start_i])
            end_x = float(x[end_i])
            length = (end_x - start_x) + dt
            count = int(end_i - start_i + 1)
            ranges.append((start_x, end_x, float(length), count))
            in_run = False
    if in_run:
        end_i = len(mask) - 1
        start_x = float(x[start_i])
        end_x = float(x[end_i])
        length = (end_x - start_x) + dt
        count = int(end_i - start_i + 1)
        ranges.append((start_x, end_x, float(length), count))

    return ranges


def save_range_stats_txt(
    txt_path: str,
    *,
    x_label: str,
    threshold_75: float,
    threshold_95: float,
    ranges_75: list[tuple[float, float, float, int]],
    ranges_95: list[tuple[float, float, float, int]],
) -> None:
    def avg_length(ranges: list[tuple[float, float, float, int]]) -> float:
        if not ranges:
            return 0.0
        return float(np.mean([r[2] for r in ranges]))

    avg_75 = avg_length(ranges_75)
    avg_95 = avg_length(ranges_95)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("이상치 구간(연속 범위) 통계\n")
        f.write(f"- x축 단위: {x_label}\n")
        f.write(f"- 75% 기준값(주의): {threshold_75:.6f}\n")
        f.write(f"- 95% 기준값(위험): {threshold_95:.6f}\n")
        f.write("\n")

        f.write("[75% 초과 구간]\n")
        f.write(f"- 구간 수: {len(ranges_75)}\n")
        f.write(f"- 평균 구간 길이: {avg_75:.6f}\n")
        for i, (sx, ex, length, count) in enumerate(ranges_75, start=1):
            f.write(f"  {i:03d}) start={sx:.3f}, end={ex:.3f}, length={length:.3f}, points={count}\n")
        f.write("\n")

        f.write("[95% 초과 구간]\n")
        f.write(f"- 구간 수: {len(ranges_95)}\n")
        f.write(f"- 평균 구간 길이: {avg_95:.6f}\n")
        for i, (sx, ex, length, count) in enumerate(ranges_95, start=1):
            f.write(f"  {i:03d}) start={sx:.3f}, end={ex:.3f}, length={length:.3f}, points={count}\n")

def load_raw_data(csv_path: str):
    """원본 CSV 데이터를 로드 (정규화 전)"""
    raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    # key 컬럼(0번) 제외
    features = raw[:, 1:].astype(np.float32)
    return features


def get_feature_names():
    """7개 feature 컬럼 이름 반환"""
    return [
        "Pressure",
        "Power1",
        "Power2",
        "Vibration_Peak1",
        "Vibration_RMS1",
        "Vibration_Peak2",
        "Vibration_RMS2",
    ]


def visualize_anomalies(
    csv_path: str,
    model_path: str,
    seq_len: int = 50,
    stride: int = 1,
    batch_size: int = 64,
    output_path: str = "anomaly_visualization.png",
):
    """
    이상치 탐지 결과를 시각화하여 저장합니다.

    Args:
        csv_path: CSV 데이터 경로
        model_path: 학습된 모델 파일 경로
        seq_len: 시퀀스 길이
        stride: 슬라이딩 윈도우 stride
        batch_size: 배치 크기
        output_path: 저장할 이미지 파일 경로
    """
    configure_korean_matplotlib_font()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 원본 데이터 로드 (정규화 전)
    raw_features = load_raw_data(csv_path)
    num_samples, num_features = raw_features.shape
    print(f"원본 데이터: {num_samples} 샘플, {num_features} features")

    # Dataset & DataLoader (정규화된 데이터로 평가)
    dataset = SensorSequenceDataset(
        csv_path=csv_path,
        seq_len=seq_len,
        stride=stride,
        normalize=True,
    )
    input_dim = dataset.input_dim
    print(f"시퀀스 데이터셋 크기: {len(dataset)}, seq_len={seq_len}, input_dim={input_dim}")

    # 모델 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    ckpt_seq_len = checkpoint.get("seq_len", seq_len)
    ckpt_input_dim = checkpoint.get("input_dim", input_dim)

    model = TCAD(
        seq_len=ckpt_seq_len,
        input_dim=ckpt_input_dim,
        d_model=128,
        transformer_layers=3,
        nhead=4,
        resnet_dims=[128, 128],
        decoder_dims=[128],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 평가 수행
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred_labels, scores, threshold_75, threshold_95 = evaluate(model, dataloader, device)

    normal_count = np.sum(pred_labels == 0)
    warning_count = np.sum(pred_labels == 1)
    danger_count = np.sum(pred_labels == 2)
    
    print(f"Anomaly score thresholds:")
    print(f"  - 주의 단계 (75th percentile): {threshold_75:.6f}")
    print(f"  - 위험 단계 (95th percentile): {threshold_95:.6f}")
    print(f"총 시퀀스 수: {len(scores)}")
    print(f"  - 정상: {normal_count} ({100 * normal_count / len(scores):.2f}%)")
    print(f"  - 주의: {warning_count} ({100 * warning_count / len(scores):.2f}%)")
    print(f"  - 위험: {danger_count} ({100 * danger_count / len(scores):.2f}%)")

    # 시퀀스 인덱스를 원본 데이터 인덱스로 매핑
    # 각 시퀀스의 시작 위치를 저장
    sequence_starts = np.array(dataset._indices)
    # 각 시퀀스의 중간 지점을 시각화에 사용 (또는 마지막 지점)
    sequence_centers = sequence_starts + seq_len // 2

    # 시각화 생성
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(num_features + 2, 1, hspace=0.3)

    feature_names = get_feature_names()

    # 1. 이상치 점수 플롯 (상단)
    ax_score = fig.add_subplot(gs[0, 0])
    ax_score.plot(sequence_centers, scores, "b-", linewidth=1, alpha=0.7, label="Anomaly Score")
    ax_score.axhline(y=threshold_75, color="orange", linestyle="--", linewidth=2, label=f"주의 단계 ({threshold_75:.4f})")
    ax_score.axhline(y=threshold_95, color="r", linestyle="--", linewidth=2, label=f"위험 단계 ({threshold_95:.4f})")
    
    # 주의 구간 (75-95%)
    ax_score.fill_between(
        sequence_centers,
        0,
        scores,
        where=(pred_labels == 1),
        color="orange",
        alpha=0.3,
        label="주의 단계",
    )
    # 위험 구간 (95% 이상)
    ax_score.fill_between(
        sequence_centers,
        0,
        scores,
        where=(pred_labels == 2),
        color="red",
        alpha=0.5,
        label="위험 단계",
    )
    ax_score.set_xlabel("Time Step")
    ax_score.set_ylabel("Anomaly Score")
    ax_score.set_title("TCAD Anomaly Detection Scores", fontsize=14, fontweight="bold")
    ax_score.legend(loc="upper right")
    ax_score.grid(True, alpha=0.3)

    # 2. 각 feature별 시계열 플롯
    for i, feature_name in enumerate(feature_names):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.plot(raw_features[:, i], "b-", linewidth=1, alpha=0.7, label=feature_name)

        # 이상치로 탐지된 구간 강조 (주의/위험 단계별)
        warning_mask = np.zeros(num_samples, dtype=bool)
        danger_mask = np.zeros(num_samples, dtype=bool)
        for seq_idx, start_idx in enumerate(sequence_starts):
            if pred_labels[seq_idx] == 1:  # 주의 단계
                end_idx = min(start_idx + seq_len, num_samples)
                warning_mask[start_idx:end_idx] = True
            elif pred_labels[seq_idx] == 2:  # 위험 단계
                end_idx = min(start_idx + seq_len, num_samples)
                danger_mask[start_idx:end_idx] = True

        # 주의 구간 배경 강조
        ax.fill_between(
            range(num_samples),
            raw_features[:, i].min(),
            raw_features[:, i].max(),
            where=warning_mask,
            color="orange",
            alpha=0.2,
            label="주의 단계" if i == 0 else "",
        )
        # 위험 구간 배경 강조
        ax.fill_between(
            range(num_samples),
            raw_features[:, i].min(),
            raw_features[:, i].max(),
            where=danger_mask,
            color="red",
            alpha=0.3,
            label="위험 단계" if i == 0 else "",
        )

        ax.set_ylabel(feature_name)
        if i == 0:
            ax.legend(loc="upper right")
        if i == len(feature_names) - 1:
            ax.set_xlabel("Time Step")
        ax.grid(True, alpha=0.3)

    # 3. 통계 요약 (하단)
    ax_stats = fig.add_subplot(gs[-1, 0])
    ax_stats.axis("off")

    stats_text = f"""
    이상치 탐지 통계 (TCAD):
    - 총 시퀀스 수: {len(scores)}
    - 정상: {normal_count} ({100 * normal_count / len(scores):.2f}%)
    - 주의 단계: {warning_count} ({100 * warning_count / len(scores):.2f}%)
    - 위험 단계: {danger_count} ({100 * danger_count / len(scores):.2f}%)
    - 주의 Threshold (75th percentile): {threshold_75:.6f}
    - 위험 Threshold (95th percentile): {threshold_95:.6f}
    - 최대 이상치 점수: {scores.max():.6f}
    - 평균 이상치 점수: {scores.mean():.6f}
    - 최소 이상치 점수: {scores.min():.6f}
    """
    # monospace는 한글 폰트가 없는 경우 깨질 수 있어 기본 폰트를 사용
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment="center")

    plt.suptitle("TCAD Anomaly Detection Visualization", fontsize=16, fontweight="bold", y=0.995)

    # 저장
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"시각화 결과를 저장했습니다: {output_path}")

    # -------------------------------
    # anomaly score 전용 차트 저장 + 구간 통계(txt) 저장
    # -------------------------------
    base, _ext = os.path.splitext(output_path)
    score_output_path = f"{base}_score.png"
    stats_output_path = f"{base}_ranges.txt"

    # score 전용 차트
    fig_score, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sequence_centers, scores, color="tab:blue", linewidth=1.0, alpha=0.85, label="Anomaly Score")
    ax.axhline(y=threshold_75, color="orange", linestyle="--", linewidth=2, label=f"75% 기준 ({threshold_75:.4f})")
    ax.axhline(y=threshold_95, color="red", linestyle="--", linewidth=2, label=f"95% 기준 ({threshold_95:.4f})")

    # 범주(정상/주의/위험) 구간을 배경으로 표시
    warn_ranges = contiguous_ranges_from_mask(sequence_centers, pred_labels == 1)
    dang_ranges = contiguous_ranges_from_mask(sequence_centers, pred_labels == 2)
    for sx, ex, _length, _count in warn_ranges:
        ax.axvspan(sx, ex, color="orange", alpha=0.12)
    for sx, ex, _length, _count in dang_ranges:
        ax.axvspan(sx, ex, color="red", alpha=0.18)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("TCAD Anomaly Score (75%/95% 기준선 및 범주)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig_score.tight_layout()
    fig_score.savefig(score_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig_score)
    print(f"anomaly score 차트를 저장했습니다: {score_output_path}")

    # 75%/95% 초과 연속 구간 평균 길이(txt)
    ranges_75 = contiguous_ranges_from_mask(sequence_centers, scores > threshold_75)
    ranges_95 = contiguous_ranges_from_mask(sequence_centers, scores > threshold_95)
    save_range_stats_txt(
        stats_output_path,
        x_label="Time Step(원본 샘플 인덱스 기준)",
        threshold_75=float(threshold_75),
        threshold_95=float(threshold_95),
        ranges_75=ranges_75,
        ranges_95=ranges_95,
    )
    print(f"구간 통계를 저장했습니다: {stats_output_path}")

    # 이상치로 탐지된 시퀀스의 시작 인덱스 출력
    warning_indices = sequence_starts[pred_labels == 1]
    danger_indices = sequence_starts[pred_labels == 2]
    print(f"\n주의 단계로 탐지된 시퀀스 시작 인덱스 (처음 10개):")
    print(warning_indices[:10] if len(warning_indices) > 0 else "없음")
    print(f"\n위험 단계로 탐지된 시퀀스 시작 인덱스 (처음 10개):")
    print(danger_indices[:10] if len(danger_indices) > 0 else "없음")


def main():
    parser = argparse.ArgumentParser(description="Visualize TCAD anomaly detection results")
    parser.add_argument(
        "--data-path",
        type=str,
        default=_default_data_path(),
        help="CSV 데이터 경로 (기본: ../data/sensor_data_rms2_fixed.csv)",
    )
    parser.add_argument("--seq-len", type=int, default=50, help="시퀀스 길이")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 stride")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument(
        "--model-path", type=str, default="tcad_model.pt", help="학습된 모델 파일 경로"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="anomaly_visualization.png",
        help="저장할 이미지 파일 경로 (기본: anomaly_visualization.png)",
    )

    args = parser.parse_args()

    visualize_anomalies(
        csv_path=args.data_path,
        model_path=args.model_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
