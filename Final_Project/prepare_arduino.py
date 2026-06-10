"""Build the arducam (deployment-domain) split -> arduino_split/.

Reads collected_arduino/<alias>/ (OV7675 captures of Asaf's plants on the demo
table) and writes a strided train/val/test tree under arduino_split/, with
canonical class folder names so labels line up with data_split/.

Strided split
-------------
The ~100 frames/class are one continuous sweep, so neighboring frames are
near-duplicates. We assign by position so train/val/test each cover the WHOLE
sweep evenly (~80/10/10): test = every 10th frame (offset 0), val = every 10th
(offset 5), rest train. Asaf confirmed the demo is at the same distance/angle, so
this correlation is acceptable; val drives early-stopping, test is the headline
"predicts the demo" number.

Folders not in COLLECTED_ALIASES (e.g. asaf_face) are ignored automatically.

Safety: collected_arduino/ is never modified. arduino_split/ is rebuilt each run.

Run:  python prepare_arduino.py   (after collecting frames, before train.py)
"""

import shutil

import config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def subset_for(i):
    """Strided assignment over sorted frames (~80/10/10, spread across sweep)."""
    if i % 10 == 0:
        return "test"
    if i % 10 == 5:
        return "val"
    return "train"


def main():
    pairs = config.alias_class_dirs(config.ARDUINO_DIR, require_all=True)

    if config.ARDUINO_SPLIT_DIR.exists():
        shutil.rmtree(config.ARDUINO_SPLIT_DIR)

    print(f"Source (read-only): {config.ARDUINO_DIR}")
    print(f"Output            : {config.ARDUINO_SPLIT_DIR}")
    print("Split             : strided ~80/10/10 (even coverage of the sweep)\n")

    print(f"{'class':<40} {'train':>6} {'val':>4} {'test':>4}")
    print("-" * 58)
    totals = {"train": 0, "val": 0, "test": 0}
    for cls, src_dir in pairs:
        files = sorted(
            p for p in src_dir.iterdir()
            if p.is_file() and not p.name.startswith(".")
            and p.suffix.lower() in IMAGE_EXTS
        )
        counts = {"train": 0, "val": 0, "test": 0}
        for i, src in enumerate(files):
            sub = subset_for(i)
            dst_dir = config.ARDUINO_SPLIT_DIR / sub / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)   # copy raw pixels, no recompress
            counts[sub] += 1
        for k in counts:
            totals[k] += counts[k]
        print(f"{cls:<40} {counts['train']:>6} {counts['val']:>4} {counts['test']:>4}")

    print("-" * 58)
    print(f"{'TOTAL':<40} {totals['train']:>6} {totals['val']:>4} {totals['test']:>4}")
    print("\nDone. arduino_split/ ready for the Stage-3 specialization.")


if __name__ == "__main__":
    main()
