
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/raw/asl_citizen/ASL_Citizen"
OUTPUT_DIR = ROOT / "data/processed"

NUM_CLASSES = 50
MIN_TRAIN = 8
MIN_VAL = 1


def find_csv(split):
    matches = [
        p for p in DATA_DIR.rglob("*.csv")
        if p.stem.lower() == split
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {split}.csv, found {matches}"
        )

    return matches[0]


def load_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {
        "Participant ID", "Video file", "Gloss"
    }

    if not required.issubset(reader.fieldnames or []):
        raise ValueError(
            f"Unexpected CSV columns: {reader.fieldnames}"
        )

    return rows


def build_video_index():
    """Find the local path of each MP4 by filename."""
    index = {}

    for path in DATA_DIR.rglob("*.mp4"):
        name = path.name

        if name in index:
            raise ValueError(
                f"Duplicate video filename: {name}"
            )

        index[name] = path.resolve()

    print(f"Found {len(index)} MP4 videos")
    return index


def main():
    rows = {
        split: load_csv(find_csv(split))
        for split in ("train", "val", "test")
    }

    video_index = build_video_index()

    def available(row):
        return Path(
            row["Video file"].strip()
        ).name in video_index

    # Class selection uses training and validation only.
    # Never select classes based on test performance.
    train_counts = Counter(
        r["Gloss"].strip()
        for r in rows["train"] if available(r)
    )

    val_counts = Counter(
        r["Gloss"].strip()
        for r in rows["val"] if available(r)
    )

    eligible = [
        gloss for gloss, count in train_counts.items()
        if count >= MIN_TRAIN
        and val_counts[gloss] >= MIN_VAL
    ]

    # Prefer signs with the most training examples.
    eligible.sort(
        key=lambda gloss: (-train_counts[gloss], gloss)
    )

    selected = sorted(eligible[:NUM_CLASSES])

    if len(selected) != NUM_CLASSES:
        raise ValueError(
            f"Only {len(selected)} eligible signs found"
        )

    label2id = {
        gloss: i for i, gloss in enumerate(selected)
    }

    id2label = {
        str(i): gloss for gloss, i in label2id.items()
    }

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    for filename, data in (
        ("label2id.json", label2id),
        ("id2label.json", id2label),
    ):
        with (OUTPUT_DIR / filename).open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=2)

    for split, split_rows in rows.items():
        samples = []

        for row in split_rows:
            gloss = row["Gloss"].strip()

            if gloss not in label2id:
                continue

            name = Path(
                row["Video file"].strip()
            ).name

            video_path = video_index.get(name)

            if video_path is None:
                continue

            samples.append({
                "video_id": Path(name).stem,
                "video_path": str(video_path),
                "gloss": gloss,
                "label": label2id[gloss],
                "signer_id": row["Participant ID"],
                "available": True,
            })

        output = OUTPUT_DIR / f"{split}.json"

        with output.open("w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)

        print(
            f"{split}: {len(samples)} videos, "
            f"{len(set(s['label'] for s in samples))} signs"
        )

    print("\nSelected 50 ASL signs:")
    for gloss, label in label2id.items():
        print(f"{label:02d}: {gloss}")

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
