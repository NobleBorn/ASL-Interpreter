
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from asl_data.asl_citizen import ASLCitizenDataset


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "data/processed/train.json"

    dataset = ASLCitizenDataset(manifest)

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    pixels = sample["pixel_values"]
    label = sample["labels"]

    print("Single video:", pixels.shape)
    print("Data type:", pixels.dtype)
    print("Label:", label.item())

    assert pixels.shape == (16, 3, 224, 224)
    assert torch.isfinite(pixels).all()

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print(
        "Batch shape:",
        batch["pixel_values"].shape
    )
    print("Labels:", batch["labels"])

    assert batch["pixel_values"].shape == (
        2, 16, 3, 224, 224
    )

    print("Dataset verification successful!")


if __name__ == "__main__":
    main()
