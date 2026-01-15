import argparse

import torch
from torch.utils.data import DataLoader

from dataset import SensorSequenceDataset
from loss import mscvae_loss
from model import MSCVAE


def _default_data_path() -> str:
    # modeling/mscvae 기준: ../data/sensor_data_rms2_fixed.csv -> modeling/data/...
    return "../data/sensor_data_rms2_fixed.csv"


def train(model, dataloader, epochs: int, device: torch.device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        num_batches = 0

        for x_batch in dataloader:
            x_batch = x_batch.to(device)  # (batch, seq_len, input_dim)

            optimizer.zero_grad()
            recon_temporal, mus, logvars = model(x_batch)

            loss, recon_loss, kl_loss = mscvae_loss(x_batch, recon_temporal, mus, logvars)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        avg_recon = running_recon / max(1, num_batches)
        avg_kl = running_kl / max(1, num_batches)
        print(f"[Epoch {epoch}/{epochs}] loss={avg_loss:.6f} (recon={avg_recon:.6f}, kl={avg_kl:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Train MSCVAE model on sensor_data_rms2_fixed.csv")
    parser.add_argument(
        "--data-path",
        type=str,
        default=_default_data_path(),
        help="CSV 데이터 경로 (기본: ../data/sensor_data_rms2_fixed.csv)",
    )
    parser.add_argument("--seq-len", type=int, default=50, help="시퀀스 길이")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 stride")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=100, help="에폭 수")
    parser.add_argument(
        "--model-path", type=str, default="mscvae_model.pt", help="학습된 모델 저장 경로"
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
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = dataset.input_dim
    print(f"Dataset size: {len(dataset)}, seq_len={args.seq_len}, input_dim={input_dim}")

    # Model 초기화
    model = MSCVAE(
        dim=input_dim,
        latent_dim=args.latent_dim,
        lstm_hidden=args.lstm_hidden,
    ).to(device)

    # 학습 수행
    train(model, dataloader, epochs=args.epochs, device=device)

    # 모델 저장
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
    print(f"Saved trained model to {args.model_path}")


if __name__ == "__main__":
    main()
