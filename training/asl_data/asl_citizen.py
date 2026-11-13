
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor


MODEL_NAME = "MCG-NJU/videomae-base"
NUM_FRAMES = 16


def sample_frames(video_path, num_frames=NUM_FRAMES):
    """
    Decode a video and uniformly sample frames.

    Returns:
        List of RGB numpy arrays.
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frames = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            # OpenCV uses BGR; VideoMAE expects RGB.
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(rgb)

    finally:
        capture.release()

    if not frames:
        raise RuntimeError(
            f"No frames decoded: {video_path}"
        )

    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames
    ).astype(np.int64)

    return [frames[index] for index in indices]


class ASLCitizenDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        processor=None,
        num_frames=NUM_FRAMES,
    ):
        self.manifest_path = Path(manifest_path)

        with self.manifest_path.open(
            "r", encoding="utf-8"
        ) as file:
            self.samples = json.load(file)

        if not self.samples:
            raise ValueError(
                f"Empty dataset: {manifest_path}"
            )

        self.processor = processor or (
            VideoMAEImageProcessor.from_pretrained(
                MODEL_NAME
            )
        )

        self.num_frames = num_frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        video_path = Path(sample["video_path"])

        if not video_path.is_file():
            raise FileNotFoundError(video_path)

        frames = sample_frames(
            video_path,
            self.num_frames
        )

        processed = self.processor(
            frames,
            return_tensors="pt"
        )

        pixel_values = processed[
            "pixel_values"
        ].squeeze(0)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return {
            "pixel_values": pixel_values,
            "labels": label,
        }
