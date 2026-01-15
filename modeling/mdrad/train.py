import argparse

import torch
from torch.utils.data import DataLoader

from dataset import SensorSequenceDataset
from loss import mdrad_loss
from model import MDRAD


def _default_data_path() -> str:
    # modeling/mdrad 기준: ../data/... -> modeling/data/...
    return "../data/sensor_data_rms2_fixed.csv"


def train(model, dataloader, epochs: int, device: torch.device, *, alpha_corr: float, beta_disc: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        running_total = 0.0
        running_x = 0.0
        running_m = 0.0
        running_d = 0.0
        num_batches = 0

        for x_batch in dataloader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()

            x_hat, m_hat, m_avg, z1, z2, z3, _z, _w = model(x_batch)
            total, recon_x, recon_m, disc = mdrad_loss(
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
            total.backward()
            optimizer.step()

            running_total += total.item()
            running_x += recon_x.item()
            running_m += recon_m.item()
            running_d += disc.item()
            num_batches += 1

        denom = max(1, num_batches)
        print(
            f"[Epoch {epoch}/{epochs}] total={running_total/denom:.6f} "
            f"(raw={running_x/denom:.6f}, corr={running_m/denom:.6f}, disc={running_d/denom:.6f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Train MDRAD model on sensor_data_rms2_fixed.csv")
    parser.add_argument("--data-path", type=str, default=_default_data_path(), help="CSV 데이터 경로 (기본: ../data/...)")
    parser.add_argument("--seq-len", type=int, default=50, help="시퀀스 길이")
    parser.add_argument("--stride", type=int, default=1, help="슬라이딩 윈도우 stride")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=100, help="에폭 수")
    parser.add_argument("--model-path", type=str, default="mdrad_model.pt", help="모델 저장 경로")

    parser.add_argument("--d-model", type=int, default=128, help="공통 latent 차원(d_model)")
    parser.add_argument("--transformer-layers", type=int, default=3, help="Transformer encoder layers")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer heads")

    parser.add_argument("--alpha-corr", type=float, default=1.0, help="correlation reconstruction loss 가중치")
    parser.add_argument("--beta-disc", type=float, default=0.1, help="latent discrepancy loss 가중치")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SensorSequenceDataset(
        csv_path=args.data_path,
        seq_len=args.seq_len,
        stride=args.stride,
        normalize=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = dataset.input_dim
    model = MDRAD(
        seq_len=args.seq_len,
        input_dim=input_dim,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        nhead=args.nhead,
    ).to(device)

    train(
        model,
        dataloader,
        epochs=args.epochs,
        device=device,
        alpha_corr=args.alpha_corr,
        beta_disc=args.beta_disc,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seq_len": args.seq_len,
            "input_dim": input_dim,
            "d_model": args.d_model,
            "transformer_layers": args.transformer_layers,
            "nhead": args.nhead,
            "alpha_corr": args.alpha_corr,
            "beta_disc": args.beta_disc,
        },
        args.model_path,
    )
    print(f"Saved trained model to {args.model_path}")


if __name__ == "__main__":
    main()

