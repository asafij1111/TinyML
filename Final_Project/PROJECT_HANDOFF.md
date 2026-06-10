# EE446 TinyML Final Project — Plant Detector: Full Handoff

> This document is the single source of truth for the project. It captures every
> decision, constraint, and design rationale discussed so far so that work can
> resume from scratch. Assume all previously written Python is being deleted and
> rebuilt. Read this top to bottom before writing code.

---

## 1. What the project is

A **TinyML houseplant classifier** that runs entirely on an **Arduino Nano 33 BLE
Sense** with an attached camera. It takes a live camera frame, runs on-device
inference, and outputs the predicted plant species (with a confidence value).
This is the EE446 (TinyML) final project for Asaf (asafij@uw.edu), UW.

The original proposal scored **96/100**. The 4-point deduction was: *"the
evaluation metrics are mostly qualitative and should more explicitly include
accuracy, latency, memory use, and model size."* Recovering those points is a
primary design driver — see §8.

### Professor's four improvement suggestions (must be addressed)
1. Define a smaller target plant subset early; report number of classes and
   images per class.
2. Start with a very small CNN or MobileNetV2 variant and compare compressed
   versions using accuracy, model size, latency, and memory usage.
3. Use a controlled test set of local plant images (not just neighborhood
   testing) for consistent evaluation.
4. Use transfer learning first, then compare against training from scratch under
   TinyML constraints.

---

## 2. Hardware and platform

- **Board:** Arduino Nano 33 BLE Sense (nRF52840). **256 KB RAM, 1 MB flash.**
- **Camera:** model TBD (Asaf will confirm). This determines on-device
  resolution, aspect ratio, and color format (RGB565 vs grayscale). The whole
  preprocessing pipeline must match whatever the camera outputs. **Do not finalize
  on-device preprocessing until the camera model is known.**
- **Deployment runtime:** **TensorFlow Lite for Microcontrollers (TFLite Micro /
  TFLM)**, hand-written Arduino sketch.

### Why TFLM and full Python (not Edge Impulse)
Edge Impulse is course-approved and was used by past projects (Bark Detection,
Vegetable Recognition). It was **deliberately rejected** for two reasons:
1. **Control:** TFLM lets Asaf own the entire firmware loop — confidence
   threshold, "unknown" handling, output format (label + score, top-2),
   multi-frame debouncing, serial vs LCD output. Edge Impulse hides this.
2. **Tooling/learning:** the Python+TFLM path is all plain code (readable,
   debuggable, explainable), which is where AI coding assistance is strongest and
   the learning outcome is better. Edge Impulse is a GUI platform where most work
   is clicks the assistant can't see.

**Known risk:** the hard, time-consuming part of TFLM is NOT the Python (training/
quantization is fast and low-risk). It is **on-device deployment** — camera
capture, converting the frame to the exact tensor the model expects, fitting the
tensor arena in 256 KB RAM, and debugging on hardware where help is weakest.
**De-risking rule: get the end-to-end firmware skeleton running on the board EARLY
with a tiny/dummy model (camera → preprocess → inference → print label, arena
fits), BEFORE perfecting model accuracy.** The classic failure is a perfect
Python model discovered too late to be undeployable.

---

## 3. The training pipeline is NOT TFLite

Critical concept: **TensorFlow Lite is inference-only — you never train in it.**
The flow is always:

```
Train/fine-tune in full TensorFlow (Keras)  →  convert to TFLite  →  INT8 quantize  →  deploy to TFLM
```

All learning happens in regular TensorFlow on the computer. TFLite/TFLM is just
the slim runtime that runs a finished model on the device.

---

## 4. Dataset decisions (and what was rejected)

### Chosen dataset: Kaggle "House Plant Species"
- 47 classes, ~14,790 images, **one folder per species** (drops straight into
  `image_dataset_from_directory` / `ImageFolder`).
- Already downloaded locally at `Final_Project/house_plant_species/`.
- Chosen because Asaf physically owns several of these plants → training data and
  deployment environment (same house, lighting, camera) match, which is exactly
  the condition that makes a tiny quantized model work in the field. Also
  satisfies professor suggestion #3 when combined with Asaf's own photos.

### Rejected: PlantNet-300K
- 306,146 images / 1,081 species, but ~32 GB and **never finished downloading**.
- Investigated the species list (`plantnet300K_species_id_2_name.json`): only 303
  genera, heavily skewed to Mediterranean herbs/succulents/orchids. **Zero
  coverage** of the target plants (no Acer/maple, Prunus/cherry, Monstera, money
  tree/Pachira, conifers, oak, rose). Wrong dataset entirely. Dropped from the
  proposal.

### Rejected: iNaturalist as bulk source
- Has the species but no clean structured download; its API returns per-species
  image URLs you'd have to script. Kept only as an optional supplement.

### Own photos
Asaf will add his own phone photos of the plants he owns. These:
- Go into the SAME class folders (or a parallel `own_photos/<class>/` dir merged in).
- Are partly **held out as a test set** so reported accuracy reflects real
  deployment, not just Kaggle photos (professor suggestion #3).

---

## 5. Final class list (6 classes)

> CHANGED: Daffodil REMOVED (it's an outdoor bulb, not a houseplant Asaf owns).
> Dumb Cane ADDED.

Exact Kaggle folder names (must match exactly):
1. `Aloe Vera`
2. `Asparagus Fern (Asparagus setaceus)`
3. `Dumb Cane (Dieffenbachia spp.)`
4. `Jade plant (Crassula ovata)`
5. `Money Tree (Pachira aquatica)`
6. `Monstera Deliciosa (Monstera deliciosa)`

### Why 6 classes
Decision rationale (intuition-based, validated):
- **TinyML cost is nearly flat in class count.** Peak RAM and latency are driven
  by input resolution and the backbone, NOT the number of output classes. More
  classes only widens the final dense layer (a few KB of flash). So DON'T limit
  classes to save memory.
- The real cap is **data + accuracy** (more classes → more confusion, more photos
  needed) and **collection effort** (plants Asaf can photograph well).
- **Impressiveness** plateaus past ~6; 3–4 looks trivial, 6–8 reads as a real
  classifier, >10 adds effort/accuracy risk for little "wow."
- **Visual distinctness > raw count** — picked species with distinct leaf
  shape/color.

Approximate Kaggle image counts (before adding own photos): Monstera ~547,
Money Tree ~359, Jade ~353, Aloe Vera ~252, Asparagus Fern ~169. (Dumb Cane count
to be confirmed.) Imbalance (~169 vs ~547) → handle with **class weights**, not by
discarding images.

---

## 6. Dataset quality gotcha (already discovered — must handle)

The Kaggle set hides **WebP and MPO files behind `.jpg`/`.png` extensions**
(~134 WebP + 5 MPO found across the full set). TensorFlow's image loader only
decodes JPEG/PNG/GIF/BMP and crashes with *"Unknown image file format"* on the
first WebP — even though Pillow reads them fine.

**Required cleaning step before any training:** open every image with Pillow,
`convert("RGB")`, re-save as a real `.jpg`, delete the mislabeled original. Truly
unreadable files are reported and skipped. This must run before the train/val/test
split. (PIL's `verify()` is too lenient to catch these — check `im.format`.)

---

## 7. Preprocessing — matching the Arduino camera

**Golden rule: train on images that look like what the camera feeds the model at
inference.** Crisp 12 MP phone/Kaggle photos vs a grainy 96×96 camera crop will
make the model overfit to "crispness" and fail on the board.

Four things to match (final values pending camera model):
1. **Resolution:** model input small + square. Default **96×96×3** (proven on this
   board for MobileNetV2 α=0.35). Fall back to **64×64** if the tensor arena won't
   fit in RAM.
2. **Aspect ratio:** **center-crop to square, THEN resize.** Phone photos are 4:3;
   squashing distorts. (`image_dataset_from_directory(..., crop_to_aspect_ratio=True)`
   does center-crop + resize when target is square.)
3. **Field of view / framing:** when shooting own photos, frame the plant the way
   the camera will see it (distance, how much fills the frame).
4. **Color format:** many of these cameras output RGB565 or grayscale. If
   deploying in grayscale, train in grayscale. Confirm from camera model.

**Bridge the quality gap with augmentation that DEGRADES toward the camera:** mild
Gaussian blur/noise, brightness/contrast jitter, small rotations, flips.

**Do NOT pre-resize image files on disk.** Keep originals at full quality; do
crop→resize→color-convert→augment at load time in the data pipeline, so input size
(96 vs 64) and color space can change without re-processing files.

**Normalization baked into the model:** add a `Rescaling(1/127.5, offset=-1)` layer
as the model's first compute step so the model's INPUT is raw 0–255 pixels —
exactly what the camera produces. This keeps on-device preprocessing trivial and
makes INT8 quantization line up with the hardware feed.

---

## 8. Model architecture and training

### Transfer learning (course-prescribed path)
- Backbone: **MobileNetV2, alpha (width multiplier) = 0.35** (smallest variant
  with ImageNet weights; 96×96 input is supported), `include_top=False`,
  `weights="imagenet"`.
- Head: `GlobalAveragePooling2D → Dropout(0.3) → Dense(6, softmax)`.
- **Stage 1:** freeze backbone, train head (Adam, lr 1e-3, ~20 epochs).
- **Stage 2 (fine-tune):** unfreeze top ~30 backbone layers, recompile at low lr
  (1e-5), train ~10 more epochs. This is what "fine-tuning with own photos" means
  — own photos are just part of the training set; no special mechanism.
- Loss: sparse categorical crossentropy. Use **class weights** for imbalance.
- Watch `val_accuracy`: 0.85+ is good; stalling low usually means augmentation too
  aggressive or too few epochs.

### Why transfer learning resolved the original worry
Asaf worried a from-scratch CNN wouldn't be good enough and leaned toward transfer
learning. Two independent decisions were separated: HOW you train (transfer
learning) vs WHAT you deploy (reduced MobileNetV2 via alpha + small input). The
reduced-alpha MobileNetV2 is what makes it fit in 256 KB RAM.

---

## 9. The four required metrics (recovers the −4 points)

Build a **comparison table** across model variants, each reporting all four:

| variant | accuracy | model size | latency | peak RAM |
|---|---|---|---|---|
| float baseline (Keras/TFLite float) | | | | |
| **INT8 quantized** (deployed) | | | | |
| pruned | | | | |
| from-scratch small CNN | | | | |

- **INT8 quantization:** full-integer, needs a **representative dataset** (~300 real
  training images) so the converter picks quantization ranges. Keep input/output
  INT8 to match the MCU.
- **Latency / peak RAM:** Python-side numbers are host estimates for *relative*
  comparison only. **Real on-device latency and true tensor-arena size come from
  the Arduino sketch** (report what TFLM prints after `AllocateTensors()`).
- Embed the model in firmware via `xxd -i model_int8.tflite > model_int8.h`.

---

## 10. Timeline / logistics

- Course deadlines: presentations **Thu June 4** OR **Fri June 12** (4:30–6:20 PM PT);
  report due **June 12**. Today's baseline in conversation: early June 2026.
- Asaf is assigned **June 4** but can move it if needed; **he manages his own time
  — do not nag about scheduling.** Just build.
- Environment: a **venv** named `tinyml-arduino` created by the course setup script
  (`tinyml_env_setup_package/`), located at `~/ai/projects/tinyml-arduino`
  (NOT inside the project folder). Activate with
  `source ~/ai/projects/tinyml-arduino/bin/activate`. It's the same env as the
  Jupyter kernel "Python (tinyml-arduino)"; it has TensorFlow. Python 3.11.

---

## 11. Step-by-step plan for the rebuild

Suggested clean structure (all under `Final_Project/`):

```
config.py                # all settings: classes, IMG_SIZE, alpha, epochs, paths
clean_data.py            # WebP/MPO -> RGB JPEG normalization (run FIRST)
prepare_data.py          # stratified train/val/test split + own-photos holdout
train.py                 # transfer learning + fine-tune, saves float Keras model
convert_and_evaluate.py  # float + INT8 TFLite, prints the 4-metric table
firmware/                # Arduino TFLM sketch (built once camera model is known)
PROJECT_HANDOFF.md       # this file
```

Order of operations:
1. **Clean** the dataset (`clean_data.py`) — WebP/MPO → RGB JPEG. Run before anything.
2. **Split** (`prepare_data.py`) — stratified per-class train/val/test; reserve
   ~half of own photos to the test set. Copy (don't move); don't resize.
3. **Train** (`train.py`) — MobileNetV2 α=0.35 transfer learning + fine-tune,
   class weights, augmentation, normalization baked in. Save `plant_model.keras`
   and `labels.txt`. Print float test accuracy.
4. **Convert + measure** (`convert_and_evaluate.py`) — float + INT8 TFLite,
   representative dataset, print accuracy/size/latency/peakRAM table; emit `xxd`
   command for the header.
5. **(Comparison)** add a small from-scratch CNN and a pruned variant to fill the
   §9 table (professor suggestions #2 and #4).
6. **Firmware skeleton EARLY** — once camera model is known: camera capture →
   center-crop/resize/color-match to model input → feed INT8 tensor → softmax →
   confidence threshold + output. Prove the loop + arena fit on hardware with a
   dummy/small model before plugging in the final model.
7. **Field test** on the real plants; record on-device latency + arena size.
8. **Report + presentation** — update proposal to drop PlantNet-300K, state final
   dataset (Kaggle House Plant + own photos, 6 classes), report classes and
   images-per-class, and present the 4-metric comparison table.

### Open items still needed from Asaf
- **Camera model** (blocks final preprocessing + all firmware work).
- Own photos of the 6 owned plants (for fine-tuning + held-out test set).

---

## 12. Key principles to not forget
- TFLite/TFLM is inference-only; train in Keras.
- Class count barely affects RAM; input resolution and backbone do.
- Match training images to the camera's real output; degrade with augmentation.
- Bake normalization into the model so it eats raw 0–255 pixels.
- Handle imbalance with class weights, not by deleting data.
- Clean WebP/MPO before training.
- De-risk by getting the hardware loop working early with a dummy model.
- Every model variant must report accuracy, size, latency, peak RAM.
