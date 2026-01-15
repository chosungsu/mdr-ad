import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SensorSequenceDataset
from loss import mdrad_anomaly_score
from model import MDRAD


def _default_data_path() -> str:
    return "../data/sensor_data_rms2_fixed.csv"


def evaluate(model, dataloader, device: torch.device, *, alpha_corr: float, beta_disc: float):
    model.eval()
    scores = []

    for x_batch in dataloader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            x_hat, m_hat, m_avg, z1, z2, z3, _z, _w = model(x_batch)
            score = (
                mdrad_anomaly_score(
                    x_batch,
                    x_hat,
                    m_avg,
                    m_hat,
                    z1,
                    z2,
                    z3,
                    alpha_corr=alpha_corr,
                    beta_disc=beta_disc,
                )
                .detach()
                .cpu()
                .numpy()
            )
            scores.append(score)

    scores = np.concatenate(scores)
    threshold_75 = np.percentile(scores, 75)
    threshold_95 = np.percentile(scores, 95)

    pred_labels = np.zeros_like(scores, dtype=int)
    pred_labels[(scores > threshold_75) & (scores <= threshold_95)] = 1
    pred_labels[scores > threshold_95] = 2

    return pred_labels, scores, float(threshold_75), float(threshold_95)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MDRAD model and compute anomaly scores")
    parser.add_argument("--data-path", type=str, default=_default_data_path(), help="CSV 데이터 경로 (기본: ../data/...)")
    parser.add_argument("--seq-len", type=int, default=50, help="시퀀스 길이(학습과 동일)")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 stride")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--model-path", type=str, default="mdrad_model.pt", help="학습된 모델 파일 경로")
    parser.add_argument("--alpha-corr", type=float, default=1.0, help="correlation reconstruction 가중치")
    parser.add_argument("--beta-disc", type=float, default=0.1, help="discrepancy 가중치")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SensorSequenceDataset(
        csv_path=args.data_path,
        seq_len=args.seq_len,
        stride=args.stride,
        normalize=True,
    )
    input_dim = dataset.input_dim
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Eval dataset size: {len(dataset)}, seq_len={args.seq_len}, input_dim={input_dim}")

    # 모델 로드 (없으면 새로 학습하지는 않음)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"학습된 모델 파일이 없습니다: {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)
    ckpt_seq_len = checkpoint.get("seq_len", args.seq_len)
    ckpt_input_dim = checkpoint.get("input_dim", input_dim)
    d_model = checkpoint.get("d_model", 128)
    transformer_layers = checkpoint.get("transformer_layers", 3)
    nhead = checkpoint.get("nhead", 4)

    if ckpt_seq_len != args.seq_len:
        print(f"[경고] 체크포인트 seq_len={ckpt_seq_len}, 현재 인자 seq_len={args.seq_len}")
    if ckpt_input_dim != input_dim:
        print(f"[경고] 체크포인트 input_dim={ckpt_input_dim}, 현재 데이터 input_dim={input_dim}")

    model = MDRAD(
        seq_len=ckpt_seq_len,
        input_dim=ckpt_input_dim,
        d_model=d_model,
        transformer_layers=transformer_layers,
        nhead=nhead,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_labels, scores, threshold_75, threshold_95 = evaluate(
        model,
        dataloader,
        device,
        alpha_corr=args.alpha_corr,
        beta_disc=args.beta_disc,
    )

    normal_count = int(np.sum(pred_labels == 0))
    warning_count = int(np.sum(pred_labels == 1))
    danger_count = int(np.sum(pred_labels == 2))

    print("Anomaly score thresholds:")
    print(f"  - 주의 단계 (75th percentile): {threshold_75:.6f}")
    print(f"  - 위험 단계 (95th percentile): {threshold_95:.6f}")
    print(f"총 시퀀스 수: {len(scores)}")
    print(f"  - 정상: {normal_count} ({100 * normal_count / len(scores):.2f}%)")
    print(f"  - 주의: {warning_count} ({100 * warning_count / len(scores):.2f}%)")
    print(f"  - 위험: {danger_count} ({100 * danger_count / len(scores):.2f}%)")


if __name__ == "__main__":
    main()

