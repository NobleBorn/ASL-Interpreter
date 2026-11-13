
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

from asl_data.asl_citizen import ASLCitizenDataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "MCG-NJU/videomae-base"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Safe Windows starting point
        pin_memory=True,
    )


def train_epoch(
    model, loader, optimizer, device,
    accumulation_steps, use_bf16
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(loader)

    for step, batch in enumerate(loader):
        pixels = batch["pixel_values"].to(
            device, non_blocking=True
        )
        labels = batch["labels"].to(
            device, non_blocking=True
        )

        # Correctly scale even the final partial group.
        group_start = (
            step // accumulation_steps
        ) * accumulation_steps

        group_size = min(
            accumulation_steps,
            num_batches - group_start
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            outputs = model(
                pixel_values=pixels,
                labels=labels,
            )

            loss = outputs.loss

        (loss / group_size).backward()

        should_update = (
            (step + 1) % accumulation_steps == 0
            or step + 1 == num_batches
        )

        if should_update:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (
            outputs.logits.argmax(dim=-1) == labels
        ).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, device, use_bf16):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        pixels = batch["pixel_values"].to(
            device, non_blocking=True
        )
        labels = batch["labels"].to(
            device, non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            outputs = model(
                pixel_values=pixels,
                labels=labels,
            )

        batch_size = labels.size(0)

        total_loss += outputs.loss.item() * batch_size
        correct += (
            outputs.logits.argmax(dim=-1) == labels
        ).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()

    if not use_bf16:
        raise RuntimeError(
            "BF16 is unavailable. Verify your CUDA build."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Mixed precision: BF16")

    set_seed()

    data_dir = ROOT / "data/processed"
    output_dir = ROOT / "checkpoints/videomae-asl50"

    output_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "label2id.json").open(
        "r", encoding="utf-8"
    ) as f:
        label2id = json.load(f)

    id2label = {
        value: key
        for key, value in label2id.items()
    }

    if len(label2id) != 50:
        raise ValueError("Expected exactly 50 labels.")

    processor = VideoMAEImageProcessor.from_pretrained(
        MODEL_NAME
    )

    train_dataset = ASLCitizenDataset(
        data_dir / "train.json",
        processor=processor,
    )

    val_dataset = ASLCitizenDataset(
        data_dir / "val.json",
        processor=processor,
    )

    print("Training videos:", len(train_dataset))
    print("Validation videos:", len(val_dataset))

    if args.smoke_test:
        train_dataset = Subset(
            train_dataset,
            range(min(4, len(train_dataset))),
        )
        val_dataset = Subset(
            val_dataset,
            range(min(4, len(val_dataset))),
        )
        args.epochs = 1
        args.accumulation = 1

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        shuffle=False,
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=50,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
    )

    best_accuracy = -1.0
    history = []

    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer,
            device, args.accumulation, use_bf16
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, device, use_bf16
        )

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        history.append(metrics)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss={train_loss:.4f} | "
            f"train acc={train_accuracy:.2%} | "
            f"val loss={val_loss:.4f} | "
            f"val acc={val_accuracy:.2%}",
            flush=True,
        )

        if (
            not args.smoke_test
            and val_accuracy > best_accuracy
        ):
            best_accuracy = val_accuracy
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print("Saved best checkpoint.")

        if not args.smoke_test:
            with (output_dir / "history.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(history, f, indent=2)

    if args.smoke_test:
        print("Smoke test completed successfully.")
    else:
        print("Training completed.")
        print("Best validation accuracy:", best_accuracy)


if __name__ == "__main__":
    main()
