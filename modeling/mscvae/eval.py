import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SensorSequenceDataset
from loss import mscvae_loss
from model import MSCVAE
from train import train as train_mscvae
from utils import attribute_matrix


def _default_data_path() -> str:
    # modeling/mscvae 기준: ../data/sensor_data_rms2_fixed.csv -> modeling/data/...
    return "../data/sensor_data_rms2_fixed.csv"


def anomaly_score(x, recon_temporal):
    """
    이상치 점수 계산
    
    Args:
        x: 원본 입력 (batch, seq, dim)
        recon_temporal: 재구성된 attribute matrix (batch, seq, 1, dim, dim)
    
    Returns:
        scores: (batch,) 이상치 점수
    """
    M = attribute_matrix(x)  # (batch, seq, dim, dim)
    M = M.unsqueeze(2)  # (batch, seq, 1, dim, dim)
    
    # Reconstruction error를 이상치 점수로 사용
    recon_err = torch.mean((M - recon_temporal) ** 2, dim=[2, 3, 4])  # (batch, seq)
    # 시퀀스 전체의 평균
    scores = torch.mean(recon_err, dim=1)  # (batch,)
    return scores


def evaluate(model, dataloader, device: torch.device):
    model.eval()
    scores = []

    for x_batch in dataloader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            recon_temporal, mus, logvars = model(x_batch)
            score = anomaly_score(x_batch, recon_temporal).cpu().numpy()
            scores.append(score)

    scores = np.concatenate(scores)
    threshold_75 = np.percentile(scores, 75)
    threshold_95 = np.percentile(scores, 95)
    
    # 4단계 분류: 0=정상, 1=주의, 2=위험
    pred_labels = np.zeros_like(scores, dtype=int)
    pred_labels[(scores > threshold_75) & (scores <= threshold_95)] = 1  # 주의
    pred_labels[scores > threshold_95] = 2  # 위험
    
    return pred_labels, scores, threshold_75, threshold_95


def main():
    parser = argparse.ArgumentParser(description="Evaluate MSCVAE model and compute anomaly scores")
    parser.add_argument(
        "--data-path",
        type=str,
        default=_default_data_path(),
        help="CSV 데이터 경로 (기본: ../data/sensor_data_rms2_fixed.csv)",
    )
    parser.add_argument("--seq-len", type=int, default=50, help="시퀀스 길이 (학습 시 사용값과 동일해야 함)")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 stride")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument(
        "--model-path", type=str, default="mscvae_model.pt", help="학습된 모델 파일 경로"
    )
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--lstm-hidden", type=int, default=64, help="LSTM hidden dimension")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = SensorSequenceDataset(
        csv_path=args.data_path,
        seq_len=args.seq_len,
        stride=args.stride,
        normalize=True,
    )
    input_dim = dataset.input_dim
    print(f"Eval dataset size: {len(dataset)}, seq_len={args.seq_len}, input_dim={input_dim}")

    # -------------------------------
    # 1) 모델 체크포인트 존재 여부 및 입력 차원/시퀀스 길이 확인
    #    - 없거나, 현재 데이터와 차원이 다르면 새로 학습
    # -------------------------------
    need_retrain = False
    ckpt_seq_len = args.seq_len
    ckpt_input_dim = input_dim
    ckpt_latent_dim = args.latent_dim
    ckpt_lstm_hidden = args.lstm_hidden

    if not os.path.exists(args.model_path):
        print(f"학습된 모델 파일이 없어 새로 학습을 시작합니다: {args.model_path}")
        need_retrain = True
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        ckpt_seq_len = checkpoint.get("seq_len", args.seq_len)
        ckpt_input_dim = checkpoint.get("input_dim", input_dim)
        ckpt_latent_dim = checkpoint.get("latent_dim", args.latent_dim)
        ckpt_lstm_hidden = checkpoint.get("lstm_hidden", args.lstm_hidden)

        if ckpt_seq_len != args.seq_len:
            print(f"[경고] 체크포인트 seq_len={ckpt_seq_len}, 현재 인자 seq_len={args.seq_len}")
        if ckpt_input_dim != input_dim:
            print(f"[경고] 체크포인트 input_dim={ckpt_input_dim}, 현재 데이터 input_dim={input_dim}")
            print("입력 차원이 달라 기존 체크포인트를 사용할 수 없어 새로 학습합니다.")
            need_retrain = True

    if need_retrain:
        # 학습용 DataLoader (shuffle=True)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = MSCVAE(
            dim=input_dim,
            latent_dim=args.latent_dim,
            lstm_hidden=args.lstm_hidden,
        ).to(device)

        # 기본 100 epoch 학습 (필요시 인자 추가로 조정 가능)
        train_mscvae(model, train_loader, epochs=100, device=device)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "seq_len": args.seq_len,
                "input_dim": input_dim,
                "latent_dim": args.latent_dim,
                "lstm_hidden": args.lstm_hidden,
            },
            args.model_path,
        )
        print(f"새로 학습된 모델을 저장했습니다: {args.model_path}")
        ckpt_seq_len = args.seq_len
        ckpt_input_dim = input_dim
        ckpt_latent_dim = args.latent_dim
        ckpt_lstm_hidden = args.lstm_hidden
    else:
        model = MSCVAE(
            dim=ckpt_input_dim,
            latent_dim=ckpt_latent_dim,
            lstm_hidden=ckpt_lstm_hidden,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # 평가용 DataLoader (shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 평가 수행
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


if __name__ == "__main__":
    main()
