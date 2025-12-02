# src/trainers/train_gcl.py
from __future__ import annotations

import argparse

import torch
from torch.optim import Adam

from wl_gcl.src.data_loader import load_dataset
from wl_gcl.src.models.base_gnn import BaseGCN
from wl_gcl.src.contrastive.losses import info_nce_loss, nt_xent_loss


def train(args):

    dataset = load_dataset(args.dataset)
    data = dataset.data

    x = data.x.to(args.device)
    edge_index = data.edge_index.to(args.device)


    model = BaseGCN(
        in_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr)


    for epoch in range(1, args.epochs + 1):

        model.train()
        optimizer.zero_grad()

        # Two "views" — currently identical placeholder
        z1 = model(x, edge_index)
        z2 = model(x, edge_index)

        loss = nt_xent_loss(z1, z2, temperature=args.temperature)

        loss.backward()
        optimizer.step()

        if epoch % args.log_interval == 0:
            print(f"[Epoch {epoch:03d}] Loss = {loss.item():.4f}")

    print("Training complete.")



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
