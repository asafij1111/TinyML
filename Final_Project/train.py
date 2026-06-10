"""Train the houseplant classifier: general features (Kaggle+phone) -> arducam domain.

Strategy (see the design discussion / PROJECT_HANDOFF):
  Stage 1  head training      - freeze MobileNetV1 backbone, train new head on the
                                MAIN dataset (Kaggle+phone), DEGRADE-augmented so
                                the crisp images look more like the OV7675 feed.
  Stage 2  backbone fine-tune - unfreeze top FT_UNFREEZE layers, low LR, main data.
                                -> BASELINE: main test + arducam test (domain gap)
                                -> save models/plant_model_main.keras
  Stage 3  arducam specialize - adapt to the DEPLOYMENT domain WITHOUT forgetting
                                the MobileNet/Kaggle features:
                                  * freeze early/mid backbone; unfreeze only the
                                    top DOMAIN_UNFREEZE layers + head
                                  * low LR, early-stop on arducam VAL
                                  * REHEARSAL: mix a fraction of main data into
                                    each batch to anchor general features
                                -> headline: arducam TEST accuracy (predicts demo)
                                -> save models/plant_model.keras + labels.txt

Normalization (0..255 -> -1..1) is baked into the model's first layer, so the
model eats raw 0..255 pixels -- exactly what the camera produces.

Run:  python train.py     (needs data_split/ and arduino_split/)
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF C++ info/warnings
import warnings
warnings.filterwarnings("ignore")

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
from tensorflow.keras import layers, models

tf.get_logger().setLevel("ERROR")

import config

AUTOTUNE = tf.data.AUTOTUNE
IMG_SHAPE = (config.IMG_SIZE, config.IMG_SIZE, config.IMG_CHANNELS)


# --------------------------------------------------------------------------
# Datasets (all return UNBATCHED (image[0..255], int label) in canonical order)
# --------------------------------------------------------------------------
def main_ds(subset, classes):
    return tf.keras.utils.image_dataset_from_directory(
        config.SPLIT_DIR / subset,
        labels="inferred", label_mode="int", class_names=classes,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        crop_to_aspect_ratio=True,          # center-crop square, then resize
        batch_size=None, shuffle=(subset == "train"),
    )


def arduino_ds(subset, classes):
    """Explicit per-class file list so labels match canonical class order."""
    base = config.ARDUINO_SPLIT_DIR / subset
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found -- run prepare_arduino.py first")
    exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    paths, labels = [], []
    for label, cls in enumerate(classes):
        files = []
        for pat in exts:
            files += [str(p) for p in (base / cls).glob(pat)]
        for f in sorted(set(files)):
            paths.append(f)
            labels.append(label)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                                 expand_animations=False)
        side = tf.reduce_min(tf.shape(img)[:2])
        img = tf.image.resize_with_crop_or_pad(img, side, side)   # center square
        img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if subset == "train":
        ds = ds.shuffle(len(paths), seed=config.SEED)
    return ds.map(load, num_parallel_calls=AUTOTUNE)


# --------------------------------------------------------------------------
# Augmentation (operate on a single 0..255 image)
# --------------------------------------------------------------------------
def geo_augment(img, label):
    img = tf.image.random_flip_left_right(img)
    return img, label


def degrade_augment(img, label):
    """Main-dataset augmentation.

    PROBABILISTIC: every image gets light photometric jitter (so the model keeps
    learning crisp, robust features), and ~50% additionally get a gentle
    camera-like degrade (color cast, low contrast, soft focus, sensor noise) so
    the model also sees OV7675-like inputs. Applying the heavy degrade to only
    half the images keeps the general base strong instead of training purely on
    degraded inputs (which previously dropped main + baseline-arducam accuracy)."""
    img, label = geo_augment(img, label)
    x = img / 255.0
    # always-on light jitter
    x = tf.image.random_brightness(x, 0.12)
    x = tf.image.random_contrast(x, 0.8, 1.1)
    x = tf.clip_by_value(x, 0.0, 1.0)

    def heavy(z):
        z = tf.image.random_hue(z, 0.03)            # magenta/green cast
        z = tf.image.random_saturation(z, 0.75, 1.25)
        z = tf.image.random_contrast(z, 0.6, 1.0)
        soft = config.IMG_SIZE * 2 // 3             # downscale to 64, then back
        z = tf.image.resize(tf.image.resize(z, (soft, soft)),
                            (config.IMG_SIZE, config.IMG_SIZE))
        z = z + tf.random.normal(tf.shape(z), stddev=0.015)
        return tf.clip_by_value(z, 0.0, 1.0)

    x = tf.cond(tf.random.uniform([]) < 0.5, lambda: heavy(x), lambda: x)
    return x * 255.0, label


def arducam_augment(img, label):
    """Light augmentation for arducam (already the target domain)."""
    img, label = geo_augment(img, label)
    x = img / 255.0
    x = tf.image.random_brightness(x, 0.08)
    x = tf.image.random_contrast(x, 0.85, 1.15)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x * 255.0, label


def batched(ds, augment=None, shuffle_buf=0):
    if shuffle_buf:
        ds = ds.shuffle(shuffle_buf, seed=config.SEED)
    if augment is not None:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(config.BATCH_SIZE).prefetch(AUTOTUNE)


def class_weights(train_ds, n):
    counts = [0] * n
    for _, y in train_ds:
        counts[int(y)] += 1
    total = sum(counts)
    return {i: total / (n * c) if c else 0.0 for i, c in enumerate(counts)}, counts


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(n):
    inputs = layers.Input(shape=IMG_SHAPE)                  # raw 0..255 pixels
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs) # -> -1..1, baked in
    # MobileNetV1 (not V2): no inverted-residual channel expansion, so its peak
    # activation buffers are small enough to fit the 256KB RAM at 96x96.
    backbone = tf.keras.applications.MobileNet(
        input_shape=IMG_SHAPE, alpha=config.ALPHA,
        include_top=False, weights="imagenet")
    backbone.trainable = False
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(config.DROPOUT)(x)
    outputs = layers.Dense(n, activation="softmax")(x)
    return models.Model(inputs, outputs, name="plant_classifier"), backbone


def set_trainable_top(backbone, n_top):
    """Unfreeze only the top n_top backbone layers (protect general features)."""
    backbone.trainable = True
    for layer in backbone.layers[:-n_top]:
        layer.trainable = False


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def acc(model, ds):
    return model.evaluate(ds, verbose=0)[1]


def main_early():
    """Fresh early-stopping callback for the main stages (keeps best weights)."""
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.MAIN_PATIENCE,
        restore_best_weights=True)


def fmt_stage(name, h):
    """Compact per-stage line: epochs, train-acc trend, val-acc trend + best
    epoch, and val-loss trend (rising val_loss = overfitting / domain mismatch)."""
    hi = h.history
    tr = hi.get("accuracy", [float("nan")])
    va = hi.get("val_accuracy", [float("nan")])
    vl = hi.get("val_loss", [float("nan")])
    best = max(range(len(va)), key=lambda i: va[i])
    return (f"{name:<17} {len(va):>3}  {tr[0]:.2f}->{tr[-1]:.2f}   "
            f"{va[0]:.2f}->{va[-1]:.2f} (best {va[best]:.2f}@{best + 1})   "
            f"{vl[0]:.2f}->{vl[-1]:.2f} (min {min(vl):.2f})")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    classes = config.get_classes()
    n = len(classes)
    config.MODEL_DIR.mkdir(exist_ok=True)

    # ---- Main dataset (Kaggle + phone) ----
    main_train_raw = main_ds("train", classes)
    main_val = batched(main_ds("val", classes))
    main_test = batched(main_ds("test", classes))
    cw, counts = class_weights(main_train_raw, n)
    main_train = batched(main_train_raw, augment=degrade_augment)
    print("Main-dataset class counts (train):")
    for i, c in enumerate(classes):
        print(f"  {c:<40} n={counts[i]:>5}  w={cw[i]:.2f}")

    # ---- Arducam (deployment) splits ----
    ard_train_raw = arduino_ds("train", classes)
    ard_val = batched(arduino_ds("val", classes))
    ard_test = batched(arduino_ds("test", classes))

    model, backbone = build_model(n)

    # ---- Stage 1: head only ----
    print("\n=== Stage 1: head training (frozen backbone) ===")
    compile_model(model, config.HEAD_LR)
    h1 = model.fit(main_train, validation_data=main_val,
                   epochs=config.HEAD_EPOCHS, class_weight=cw,
                   callbacks=[main_early()])

    # ---- Stage 2: fine-tune top of backbone ----
    print("\n=== Stage 2: backbone fine-tune (main dataset) ===")
    set_trainable_top(backbone, config.FT_UNFREEZE)
    compile_model(model, config.FT_LR)
    h2 = model.fit(main_train, validation_data=main_val,
                   epochs=config.FT_EPOCHS, class_weight=cw,
                   callbacks=[main_early()])

    base_main = acc(model, main_test)
    base_ard = acc(model, ard_test)
    model.save(config.KAGGLE_MODEL)

    # ---- Stage 3: arducam specialization with rehearsal ----
    print("\n=== Stage 3: arducam specialization (freeze early/mid + rehearsal) ===")
    set_trainable_top(backbone, config.DOMAIN_UNFREEZE)
    compile_model(model, config.DOMAIN_LR)

    # rehearsal: mostly arducam, with a fraction of main data to anchor features
    ard_aug = ard_train_raw.shuffle(2000, seed=config.SEED).map(
        arducam_augment, num_parallel_calls=AUTOTUNE).repeat()
    main_aug = main_train_raw.shuffle(2000, seed=config.SEED).map(
        degrade_augment, num_parallel_calls=AUTOTUNE).repeat()
    mixed = tf.data.Dataset.sample_from_datasets(
        [ard_aug, main_aug],
        weights=[1.0 - config.REHEARSAL_WEIGHT, config.REHEARSAL_WEIGHT],
        seed=config.SEED,
    ).batch(config.BATCH_SIZE).prefetch(AUTOTUNE)

    # one "epoch" ~ a pass over the arducam train set
    n_ard_train = sum(1 for _ in ard_train_raw)
    steps = max(1, n_ard_train // config.BATCH_SIZE)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=config.EARLY_STOP_PATIENCE,
        restore_best_weights=True, mode="max")
    h3 = model.fit(mixed, validation_data=ard_val, epochs=config.DOMAIN_EPOCHS,
                   steps_per_epoch=steps, callbacks=[early])

    final_main = acc(model, main_test)
    final_ard = acc(model, ard_test)
    model.save(config.FINAL_MODEL)
    config.LABELS_FILE.write_text("\n".join(classes) + "\n")

    # ---- Summary ----
    print("\n=========================== TRAINING SUMMARY ===========================")
    print(f"{'stage':<17} {'ep':>3}  {'train_acc':<11}  "
          f"{'val_acc (best@ep)':<22} {'val_loss (min)'}")
    print(fmt_stage("S1 head (main)", h1))
    print(fmt_stage("S2 ft (main)", h2))
    print(fmt_stage("S3 domain (ard)", h3))

    print("\n=============================== RESULTS ================================")
    print(f"  arducam TEST : {base_ard:.4f} -> {final_ard:.4f}  ({final_ard - base_ard:+.4f})")
    print(f"  main TEST    : {base_main:.4f} -> {final_main:.4f}  ({final_main - base_main:+.4f})")
    print(f"  saved {config.FINAL_MODEL.name}, {config.KAGGLE_MODEL.name}, {config.LABELS_FILE.name}")


if __name__ == "__main__":
    main()
