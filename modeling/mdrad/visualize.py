import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SensorSequenceDataset
from eval import evaluate
from model import MDRAD


def _default_data_path() -> str:
    return "../data/sensor_data_rms2_fixed.csv"


def configure_korean_matplotlib_font() -> None:
    # 음수 기호 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    try:
        from matplotlib import font_manager as fm

        available = {f.name for f in fm.fontManager.ttflist}
        for name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]:
            if name in available:
                plt.rcParams["font.family"] = name
                return
    except Exception:
        plt.rcParams["font.family"] = "Malgun Gothic"


def contiguous_ranges_from_mask(x: np.ndarray, mask: np.ndarray) -> list[tuple[float, float, float, int]]:
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
            sx = float(x[start_i])
            ex = float(x[end_i])
            length = (ex - sx) + dt
            count = int(end_i - start_i + 1)
            ranges.append((sx, ex, float(length), count))
            in_run = False
    if in_run:
        end_i = len(mask) - 1
        sx = float(x[start_i])
        ex = float(x[end_i])
        length = (ex - sx) + dt
        count = int(end_i - start_i + 1)
        ranges.append((sx, ex, float(length), count))

    return ranges


def save_range_stats_txt(
    txt_path: str,
    *,
    threshold_75: float,
    threshold_95: float,
    ranges_75: list[tuple[float, float, float, int]],
    ranges_95: list[tuple[float, float, float, int]],
) -> None:
    def avg_len(r: list[tuple[float, float, float, int]]) -> float:
        return float(np.mean([x[2] for x in r])) if r else 0.0

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("이상치 구간(연속 범위) 통계 - MDRAD\n")
        f.write(f"- 75% 기준값(주의): {threshold_75:.6f}\n")
        f.write(f"- 95% 기준값(위험): {threshold_95:.6f}\n\n")

        f.write("[75% 초과 구간]\n")
        f.write(f"- 구간 수: {len(ranges_75)}\n")
        f.write(f"- 평균 구간 길이: {avg_len(ranges_75):.6f}\n")
        for i, (sx, ex, length, count) in enumerate(ranges_75, start=1):
            f.write(f"  {i:03d}) start={sx:.3f}, end={ex:.3f}, length={length:.3f}, points={count}\n")

        f.write("\n[95% 초과 구간]\n")
        f.write(f"- 구간 수: {len(ranges_95)}\n")
        f.write(f"- 평균 구간 길이: {avg_len(ranges_95):.6f}\n")
        for i, (sx, ex, length, count) in enumerate(ranges_95, start=1):
            f.write(f"  {i:03d}) start={sx:.3f}, end={ex:.3f}, length={length:.3f}, points={count}\n")


def visualize_anomalies(
    csv_path: str,
    model_path: str,
    seq_len: int = 50,
    stride: int = 1,
    batch_size: int = 64,
    alpha_corr: float = 1.0,
    beta_disc: float = 0.1,
    output_path: str = "anomaly_visualization.png",
):
    configure_korean_matplotlib_font()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SensorSequenceDataset(csv_path=csv_path, seq_len=seq_len, stride=stride, normalize=True)
    input_dim = dataset.input_dim
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    ckpt = torch.load(model_path, map_location=device)
    model = MDRAD(
        seq_len=int(ckpt.get("seq_len", seq_len)),
        input_dim=int(ckpt.get("input_dim", input_dim)),
        d_model=int(ckpt.get("d_model", 128)),
        transformer_layers=int(ckpt.get("transformer_layers", 3)),
        nhead=int(ckpt.get("nhead", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    pred_labels, scores, threshold_75, threshold_95 = evaluate(
        model, dataloader, device, alpha_corr=alpha_corr, beta_disc=beta_disc
    )

    sequence_starts = np.array(dataset._indices)
    sequence_centers = sequence_starts + seq_len // 2

    normal_count = int(np.sum(pred_labels == 0))
    warning_count = int(np.sum(pred_labels == 1))
    danger_count = int(np.sum(pred_labels == 2))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sequence_centers, scores, color="tab:blue", linewidth=1.0, alpha=0.85, label="Anomaly Score")
    ax.axhline(y=threshold_75, color="orange", linestyle="--", linewidth=2, label=f"75% 기준 ({threshold_75:.4f})")
    ax.axhline(y=threshold_95, color="red", linestyle="--", linewidth=2, label=f"95% 기준 ({threshold_95:.4f})")

    warn_ranges = contiguous_ranges_from_mask(sequence_centers, pred_labels == 1)
    dang_ranges = contiguous_ranges_from_mask(sequence_centers, pred_labels == 2)
    for sx, ex, _l, _c in warn_ranges:
        ax.axvspan(sx, ex, color="orange", alpha=0.12)
    for sx, ex, _l, _c in dang_ranges:
        ax.axvspan(sx, ex, color="red", alpha=0.18)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("MDRAD Anomaly Score (75%/95% 기준선 및 범주)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"시각화 결과를 저장했습니다: {output_path}")

    base, _ = os.path.splitext(output_path)
    stats_txt = f"{base}_ranges.txt"
    ranges_75 = contiguous_ranges_from_mask(sequence_centers, scores > threshold_75)
    ranges_95 = contiguous_ranges_from_mask(sequence_centers, scores > threshold_95)
    save_range_stats_txt(
        stats_txt,
        threshold_75=float(threshold_75),
        threshold_95=float(threshold_95),
        ranges_75=ranges_75,
        ranges_95=ranges_95,
    )
    print(f"구간 통계를 저장했습니다: {stats_txt}")

    print("요약:")
    print(f"- 총 시퀀스 수: {len(scores)}")
    print(f"- 정상: {normal_count} ({100 * normal_count / len(scores):.2f}%)")
    print(f"- 주의: {warning_count} ({100 * warning_count / len(scores):.2f}%)")
    print(f"- 위험: {danger_count} ({100 * danger_count / len(scores):.2f}%)")
    print(f"- threshold_75: {threshold_75:.6f}")
    print(f"- threshold_95: {threshold_95:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MDRAD anomaly detection results")
    parser.add_argument("--data-path", type=str, default=_default_data_path(), help="CSV 데이터 경로 (기본: ../data/...)")
    parser.add_argument("--seq-len", type=int, default=50, help="시퀀스 길이")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 stride")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--model-path", type=str, default="mdrad_model.pt", help="학습된 모델 파일 경로")
    parser.add_argument("--alpha-corr", type=float, default=1.0, help="correlation reconstruction 가중치")
    parser.add_argument("--beta-disc", type=float, default=0.1, help="discrepancy 가중치")
    parser.add_argument("--output-path", type=str, default="anomaly_visualization.png", help="저장할 이미지 파일 경로")

    args = parser.parse_args()
    visualize_anomalies(
        csv_path=args.data_path,
        model_path=args.model_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        alpha_corr=args.alpha_corr,
        beta_disc=args.beta_disc,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

