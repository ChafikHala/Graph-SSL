# main_debug.py
from __future__ import annotations

from wl_gcl.src.data.dataset import load_dataset, get_splits


DATASETS = ["Cora", "Citeseer", "Pubmed", "Actor", "Squirrel"]


def main() -> None:

    for name in DATASETS:
        print("=" * 50)
        print(f"Loading dataset: {name}")

        ds = load_dataset(name)
        data = ds.data

        train_mask, val_mask, test_mask = get_splits(data)

        print(f"Nodes:        {data.num_nodes}")
        print(f"Edges:        {data.num_edges}")
        print(f"Features:     {ds.num_features}")
        print(f"Classes:      {ds.num_classes}")
        print(f"Train size:   {int(train_mask.sum())}")
        print(f"Val size:     {int(val_mask.sum())}")
        print(f"Test size:    {int(test_mask.sum())}")
        print()


if __name__ == "__main__":
    main()
