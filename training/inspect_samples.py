
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from transformers import VideoMAEImageProcessor

from asl_data.asl_citizen import sample_frames
import random


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "MCG-NJU/videomae-base"


def main():
    manifest_path = ROOT / "data/processed/val.json"

    with manifest_path.open(encoding="utf-8") as f:
        samples = json.load(f)

    
    random.seed(42)
    sample = random.choice(samples)
    frames = sample_frames(sample["video_path"])

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_NAME
    )

    processed = processor(
        frames,
        return_tensors="pt"
    )["pixel_values"][0]

    # Reverse normalization so we can visualize the input.
    mean = np.asarray(
        processor.image_mean
    ).reshape(3, 1, 1)

    std = np.asarray(
        processor.image_std
    ).reshape(3, 1, 1)

    fig, axes = plt.subplots(
        4, 4, figsize=(14, 12)
    )

    for i, ax in enumerate(axes.flat):
        tensor = processed[i].numpy()

        image = tensor * std + mean
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)

        ax.imshow(image)
        ax.set_title(f"Frame {i}")
        ax.axis("off")

    fig.suptitle(
        f"Sign: {sample['gloss']}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
