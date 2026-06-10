"""Build the MAIN dataset split (Kaggle + phone photos) -> data_split/.

What this does
--------------
For each canonical class it gathers images from TWO read-only sources:
  - house_plant_species/<class>/        (Kaggle, generic)
  - collected_images/<alias>/           (Asaf's phone photos, merged in)
then sanitizes each (Pillow: apply EXIF rotation, convert RGB, re-encode as real
JPEG -- this neutralizes the WebP/MPO files hiding behind .jpg, and the phone
photos' sideways EXIF) and writes a stratified 70/15/15 train/val/test tree.

The phone photos are GROUPED IN as ordinary training data (they teach general
"Asaf's plants" features). The OV7675 captures are handled separately by
prepare_arduino.py -- they are the deployment-domain specialization set.

Safety: source folders are never modified. data_split/ is rebuilt from scratch
each run (idempotent); only ever deletes inside data_split/.

Run:  python prepare_data.py
"""

import random
import shutil
from collections import defaultdict

from PIL import Image, ImageOps

import config

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".bmp", ".gif", ".mpo"}
JPEG_QUALITY = 95


def list_images(folder):
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTS
    )


def sanitize_to_jpeg(src, dst):
    """Open src, apply EXIF rotation, convert RGB, save real JPEG at dst.

    Returns the true source format (e.g. "JPEG", "WEBP", "MPO") or None on
    failure. Never modifies src.
    """
    try:
        with Image.open(src) as im:
            fmt = im.format
            ImageOps.exif_transpose(im).convert("RGB").save(
                dst, "JPEG", quality=JPEG_QUALITY)
        return fmt
    except Exception as exc:
        print(f"    [skip] unreadable: {src.name} ({exc})")
        return None


def split_indices(n, ratios, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {"train": idx[:n_train],
            "val": idx[n_train:n_train + n_val],
            "test": idx[n_train + n_val:]}


def main():
    classes = config.get_classes()
    phone_map = dict(config.alias_class_dirs(config.PHONE_DIR, require_all=False))
    rng = random.Random(config.SEED)

    if config.SPLIT_DIR.exists():
        shutil.rmtree(config.SPLIT_DIR)
    for sub in ("train", "val", "test"):
        for cls in classes:
            (config.SPLIT_DIR / sub / cls).mkdir(parents=True, exist_ok=True)

    print(f"Sources (read-only): {config.RAW_DIR.name} + {config.PHONE_DIR.name}")
    print(f"Output             : {config.SPLIT_DIR}")
    print(f"Classes={len(classes)}  seed={config.SEED}  split={config.SPLIT}\n")

    stats = {cls: defaultdict(int) for cls in classes}
    fmt_counts = defaultdict(int)

    for cls in classes:
        # (file, tag) where tag identifies the source for filename prefixing
        items = [(f, "kag") for f in list_images(config.RAW_DIR / cls)]
        if cls in phone_map:
            items += [(f, "phone") for f in list_images(phone_map[cls])]
        stats[cls]["kaggle"] = sum(1 for _, t in items if t == "kag")
        stats[cls]["phone"] = sum(1 for _, t in items if t == "phone")

        parts = split_indices(len(items), config.SPLIT, rng)
        for sub, indices in parts.items():
            for j, i in enumerate(indices):
                src, tag = items[i]
                dst = config.SPLIT_DIR / sub / cls / f"{tag}_{src.stem}.jpg"
                if dst.exists():
                    dst = dst.with_name(f"{tag}_{src.stem}_{j}.jpg")
                fmt = sanitize_to_jpeg(src, dst)
                if fmt is None:
                    stats[cls]["skipped"] += 1
                    continue
                fmt_counts[fmt] += 1
                stats[cls][sub] += 1

    # ---- Report ----
    print(f"{'class':<40} {'kaggle':>6} {'phone':>5} {'skip':>4} "
          f"{'train':>6} {'val':>4} {'test':>4}")
    print("-" * 78)
    totals = defaultdict(int)
    for cls in classes:
        s = stats[cls]
        print(f"{cls:<40} {s['kaggle']:>6} {s['phone']:>5} {s['skipped']:>4} "
              f"{s['train']:>6} {s['val']:>4} {s['test']:>4}")
        for k in ("kaggle", "phone", "skipped", "train", "val", "test"):
            totals[k] += s[k]
    print("-" * 78)
    print(f"{'TOTAL':<40} {totals['kaggle']:>6} {totals['phone']:>5} "
          f"{totals['skipped']:>4} {totals['train']:>6} {totals['val']:>4} "
          f"{totals['test']:>4}")
    print("\nSource formats (all re-encoded to JPEG):")
    for fmt, n in sorted(fmt_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:<6} {n}")
    print("\nDone. data_split/ ready (run prepare_arduino.py for the domain set).")


if __name__ == "__main__":
    main()
