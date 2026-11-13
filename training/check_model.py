
import torch
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)

MODEL_NAME = "MCG-NJU/videomae-base"
NUM_CLASSES = 50


def main():
    print("Loading image processor...")

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Loading pretrained VideoMAE...")

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print("Device:", device)
    print("Number of classes:", model.config.num_labels)
    print("Frames:", model.config.num_frames)
    print("Image processor:", type(processor).__name__)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Parameters: {parameters:,}")

    # Verify the model's output shape using dummy video.
    height = model.config.image_size
    frames = model.config.num_frames

    dummy_video = torch.randn(
        1, frames, 3, height, height,
        device=device,
    )

    with torch.inference_mode():
        outputs = model(pixel_values=dummy_video)

    print("Output shape:", outputs.logits.shape)

    assert outputs.logits.shape == (1, NUM_CLASSES)

    print("Model verification successful!")


if __name__ == "__main__":
    main()
