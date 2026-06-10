"""Central configuration for the EE446 TinyML houseplant-classifier pipeline.

Data strategy (three sources)
-----------------------------
1. RAW_DIR  (house_plant_species/)  - Kaggle, generic internet photos. READ-ONLY.
2. PHONE_DIR (collected_images/)    - Asaf's plants, phone camera. Merged INTO the
   main dataset (general features). READ-ONLY.
3. ARDUINO_DIR (collected_arduino/) - Asaf's plants through the OV7675 on the demo
   table. This IS the deployment domain: used for the Stage-3 specialization and
   as the test set that predicts demo accuracy. READ-ONLY.

Hard rule: all three source folders are READ-ONLY. The pipeline only reads/copies
from them; only the GENERATED folders (data_split/, arduino_split/, models/) are
written. Classes are derived live from RAW_DIR so config never drifts from disk.
"""

from pathlib import Path

# --- Source folders (READ-ONLY) --------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "house_plant_species"   # Kaggle (generic)
PHONE_DIR = PROJECT_DIR / "collected_images"    # phone photos of Asaf's plants
ARDUINO_DIR = PROJECT_DIR / "collected_arduino" # OV7675 captures (deployment domain)

# --- Generated folders ------------------------------------------------------
SPLIT_DIR = PROJECT_DIR / "data_split"          # Kaggle + phone, train/val/test
ARDUINO_SPLIT_DIR = PROJECT_DIR / "arduino_split"  # arducam, train/val/test
MODEL_DIR = PROJECT_DIR / "models"

# Outputs
KAGGLE_MODEL = MODEL_DIR / "plant_model_main.keras"   # after stages 1-2 (general)
FINAL_MODEL = MODEL_DIR / "plant_model.keras"         # after Stage 3 (deployed)
INT8_MODEL = MODEL_DIR / "plant_model_int8.tflite"
FLOAT_MODEL = MODEL_DIR / "plant_model_float.tflite"
MODEL_HEADER = MODEL_DIR / "model_int8.h"
LABELS_FILE = MODEL_DIR / "labels.txt"

# --- Model / input ---------------------------------------------------------
IMG_SIZE = 96            # MobileNetV1 has small activations, so 96x96 fits the 256KB
                         # RAM (V2's 6x channel expansion did not). Fall back to 64 if needed.
IMG_CHANNELS = 3         # RGB (color is a strong cue for these 6 plants)
BATCH_SIZE = 32
ALPHA = 0.25             # MobileNetV1 width multiplier (small activations -> fits 96x96)
DROPOUT = 0.3

# --- Training schedule ------------------------------------------------------
# Stage 1: frozen backbone, train the new head on the main dataset.
HEAD_EPOCHS = 18
HEAD_LR = 1e-3
# Stage 2: unfreeze top FT_UNFREEZE backbone layers, fine-tune at low LR (main).
FT_EPOCHS = 10
FT_LR = 1e-5
FT_UNFREEZE = 30
# Early-stopping patience for the main stages (1-2); epochs above act as ceilings.
MAIN_PATIENCE = 4
# Stage 3: specialize on the arducam (deployment) domain WITHOUT forgetting the
# MobileNet/Kaggle features. Freeze early/mid backbone, unfreeze only the top
# DOMAIN_UNFREEZE layers + head; low LR; early-stop on arducam val; and mix a
# fraction of main data into each batch (rehearsal) to anchor general features.
DOMAIN_EPOCHS = 30
DOMAIN_LR = 1e-4
DOMAIN_UNFREEZE = 20
EARLY_STOP_PATIENCE = 6
REHEARSAL_WEIGHT = 0.25   # share of Stage-3 samples drawn from the main dataset

# --- On-device deployment measurements --------------------------------------
# These are NOT computed in Python; they are read off the Arduino at deploy time
# and recorded here so the results report can show the deployment-resource table.
#   ARENA_USED_BYTES   : interpreter->arena_used_bytes() from the boot serial
#                        print of plant_infer.ino / arena_probe.ino.
#   ARENA_ALLOC_BYTES  : kTensorArenaSize reserved in plant_infer.ino.
#   TARGET_HARDWARE    : deployment board (label for the report table).
# Update ARENA_USED_BYTES if you re-measure after changing IMG_SIZE/ALPHA.
TARGET_HARDWARE = "Arduino Nano 33 BLE Sense (nRF52840)"
ARENA_USED_BYTES = 102780      # measured on device
ARENA_ALLOC_BYTES = 112 * 1024  # reserved in plant_infer.ino

# --- Quantization -----------------------------------------------------------
# Representative dataset for full-integer INT8 calibration. Drawn from the
# arducam TRAIN split so the quantization ranges match deployment.
REP_SAMPLES = 200

# --- Splits -----------------------------------------------------------------
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}           # main dataset
ARDUINO_SPLIT = {"train": 0.80, "val": 0.10, "test": 0.10}   # arducam (strided)
SEED = 42

# Map snake_case capture folders (phone AND arducam) to canonical Kaggle classes.
# Folders not present here (e.g. a stray "asaf_face") are ignored by the prepare
# scripts, so test/scratch classes never leak into training.
COLLECTED_ALIASES = {
    "aloe_vera": "Aloe Vera",
    "asparagus_fern": "Asparagus Fern (Asparagus setaceus)",
    "dumb_cane": "Dumb Cane (Dieffenbachia spp.)",
    "jade_plant": "Jade plant (Crassula ovata)",
    "money_tree": "Money Tree (Pachira aquatica)",
    "snake_plant": "Snake plant (Sanseviera)",
}


def get_classes():
    """Canonical class names = sorted subfolders of RAW_DIR (source of truth)."""
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")
    classes = sorted(
        p.name for p in RAW_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not classes:
        raise RuntimeError(f"No class subfolders found in {RAW_DIR}")
    return classes


def alias_class_dirs(source_dir, require_all=True):
    """Map a capture folder's subdirs to canonical classes, in get_classes order.

    Returns a list of (canonical_class, subfolder_path) for every class that has
    a mapped, present subfolder under source_dir. Subfolders not in
    COLLECTED_ALIASES (e.g. asaf_face) are skipped. With require_all=True, raises
    if any canonical class is missing a folder.
    """
    classes = get_classes()
    canon_to_alias = {v: k for k, v in COLLECTED_ALIASES.items()}
    pairs = []
    for c in classes:
        alias = canon_to_alias.get(c)
        sub = source_dir / alias if alias else None
        if sub and sub.is_dir():
            pairs.append((c, sub))
        elif require_all:
            raise FileNotFoundError(
                f"Missing folder for class '{c}' under {source_dir} "
                f"(expected subfolder '{alias}')")
    return pairs


if __name__ == "__main__":
    print(f"PROJECT_DIR : {PROJECT_DIR}")
    for name, d in [("RAW", RAW_DIR), ("PHONE", PHONE_DIR), ("ARDUINO", ARDUINO_DIR)]:
        print(f"{name:8}: {d}  (exists={d.is_dir()})")
    classes = get_classes()
    print(f"CLASSES ({len(classes)}):")
    for c in classes:
        print(f"  - {c}")
