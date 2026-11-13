
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"

splits = {}

for split in ("train", "val", "test"):
    with (DATA_DIR / f"{split}.json").open(
        encoding="utf-8"
    ) as f:
        splits[split] = json.load(f)

counts = {
    split: Counter(s["gloss"] for s in samples)
    for split, samples in splits.items()
}

with (DATA_DIR / "label2id.json").open(
    encoding="utf-8"
) as f:
    label2id = json.load(f)

print(
    f"{'Sign':<25}"
    f"{'Train':>8}"
    f"{'Val':>8}"
    f"{'Test':>8}"
)

for gloss in label2id:
    print(
        f"{gloss:<25}"
        f"{counts['train'][gloss]:>8}"
        f"{counts['val'][gloss]:>8}"
        f"{counts['test'][gloss]:>8}"
    )

print("\nSummary:")
for split in counts:
    values = [
        counts[split][gloss]
        for gloss in label2id
    ]
    print(
        f"{split}: min={min(values)}, "
        f"max={max(values)}, "
        f"avg={sum(values)/len(values):.1f}"
    )
