"""Generate the Option-D results report: results/RESULTS.md + confusion-matrix PNGs.

This does NOT touch train.py / quantize.py or their terminal output. It loads the
already-trained artifacts and evaluates them on the held-out test sets, then writes
a single self-contained markdown report (tables + linked figures) for the final
report and the GitHub repo.

Evaluated models:
  - FLOAT  : models/plant_model.keras           (the "big" float model)
  - INT8   : models/plant_model_int8.tflite      (the DEPLOYED model, post-PTQ)
Evaluated on two held-out sets:
  - arducam TEST (arduino_split/test)  -> headline / demo predictor
  - main TEST    (data_split/test)     -> generalization to clean images

For each (model, test set) it reports overall accuracy, per-class
precision/recall/F1/support, macro + weighted F1, and a confusion matrix (saved
as a PNG and also embedded as a markdown table).

Dependencies are optional and degrade gracefully:
  - scikit-learn : used for the classification report if importable; otherwise a
                   pure-NumPy equivalent is computed.
  - matplotlib   : used to render confusion-matrix PNGs; if missing, the PNGs are
                   skipped and the markdown keeps the text confusion tables.

Run:  python report_metrics.py     (after train.py and quantize.py)
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import time
import warnings
warnings.filterwarnings("ignore")

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import config
from train import arduino_ds, main_ds

RESULTS_DIR = config.PROJECT_DIR / "results"

# Optional deps -------------------------------------------------------------
try:
    from sklearn.metrics import precision_recall_fscore_support
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ---------------------------------------------------------------------------
# Data + prediction helpers
# ---------------------------------------------------------------------------
def collect_numpy(ds):
    """Unbatched (img[0..255], int label) dataset -> (X float32, y int64)."""
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(int(y))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def predict_keras(model, x):
    """Class predictions from the float Keras model (eats raw 0..255 pixels)."""
    probs = model.predict(x, batch_size=config.BATCH_SIZE, verbose=0)
    return np.argmax(probs, axis=1)


def predict_tflite(tflite_path, x):
    """Class predictions from a TFLite model (handles INT8 input quantization)."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def prep(img):
        if inp["dtype"] == np.int8:
            scale, zp = inp["quantization"]
            q = np.round(img / scale) + zp
            return np.clip(q, -128, 127).astype(np.int8)
        return img.astype(np.float32)

    preds = np.empty(len(x), dtype=np.int64)
    for i, img in enumerate(x):
        interp.set_tensor(inp["index"], prep(img)[None])
        interp.invoke()
        preds[i] = int(np.argmax(interp.get_tensor(out["index"])[0]))
    return preds


def tflite_mean_latency_ms(tflite_path, x, warmup=5):
    """Mean single-frame invoke latency (ms) on the HOST, for relative comparison.
    On-device latency differs; this orders float vs INT8 and gives a ballpark."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]

    def prep(img):
        if inp["dtype"] == np.int8:
            scale, zp = inp["quantization"]
            q = np.round(img / scale) + zp
            return np.clip(q, -128, 127).astype(np.int8)
        return img.astype(np.float32)

    samples = [prep(img)[None] for img in x]
    for s in samples[:warmup]:                       # warm up (XNNPACK, caches)
        interp.set_tensor(inp["index"], s); interp.invoke()
    times = []
    for s in samples:
        interp.set_tensor(inp["index"], s)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Metrics (NumPy core; sklearn used only for the per-class P/R/F if available)
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(y_true, y_pred, n):
    """Return precision, recall, f1, support arrays (length n)."""
    if HAVE_SKLEARN:
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(n)), zero_division=0)
        return p, r, f, s
    cm = confusion(y_true, y_pred, n)
    support = cm.sum(axis=1)
    pred_tot = cm.sum(axis=0)
    diag = np.diag(cm).astype(float)
    precision = np.divide(diag, pred_tot, out=np.zeros(n), where=pred_tot > 0)
    recall = np.divide(diag, support, out=np.zeros(n), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n), where=denom > 0)
    return precision, recall, f1, support.astype(int)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def short(cls):
    """Short label for tight table headers / figure ticks."""
    return cls.split(" (")[0]


def save_confusion_png(cm, classes, title, path):
    if not HAVE_MPL:
        return False
    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.1 * n + 1.5, 1.1 * n + 1.0))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labels = [short(c) for c in classes]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > vmax / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def md_confusion_table(cm, classes):
    labels = [short(c) for c in classes]
    head = "| true \\ pred | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    lines = [head, sep]
    for i, c in enumerate(labels):
        row = " | ".join(str(int(v)) for v in cm[i])
        lines.append(f"| **{c}** | {row} |")
    return "\n".join(lines)


def md_prf_table(p, r, f, s, classes):
    lines = ["| Class | Precision | Recall | F1 | Support |",
             "|---|---|---|---|---|"]
    for i, c in enumerate(classes):
        lines.append(f"| {short(c)} | {p[i]:.3f} | {r[i]:.3f} | {f[i]:.3f} | {int(s[i])} |")
    total = int(np.sum(s))
    macro_f = float(np.mean(f))
    weighted_f = float(np.average(f, weights=s)) if total else 0.0
    lines.append(f"| **macro avg** | {np.mean(p):.3f} | {np.mean(r):.3f} | {macro_f:.3f} | {total} |")
    lines.append(f"| **weighted avg** | "
                 f"{np.average(p, weights=s) if total else 0:.3f} | "
                 f"{np.average(r, weights=s) if total else 0:.3f} | "
                 f"{weighted_f:.3f} | {total} |")
    return "\n".join(lines), macro_f, weighted_f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate(name, predict_fn, x, y, classes, set_tag, md, summary):
    n = len(classes)
    y_pred = predict_fn(x)
    acc = float(np.mean(y_pred == y))
    cm = confusion(y, y_pred, n)
    p, r, f, s = per_class_prf(y, y_pred, n)
    prf_md, macro_f, weighted_f = md_prf_table(p, r, f, s, classes)
    summary.append((name, set_tag, acc, macro_f, weighted_f))

    png_name = f"cm_{name}_{set_tag}.png".replace(" ", "_").lower()
    png_path = RESULTS_DIR / png_name
    have_png = save_confusion_png(
        cm, classes, f"{name} - {set_tag} (acc {acc:.3f})", png_path)

    md.append(f"### {name} model - {set_tag} test set\n")
    md.append(f"Overall accuracy: **{acc:.4f}**  |  macro-F1: **{macro_f:.3f}**  "
              f"|  weighted-F1: **{weighted_f:.3f}**\n")
    md.append("**Per-class metrics**\n")
    md.append(prf_md + "\n")
    md.append("**Confusion matrix** (rows = true, cols = predicted)\n")
    if have_png:
        md.append(f"![{name} {set_tag} confusion matrix]({png_name})\n")
    md.append(md_confusion_table(cm, classes) + "\n")


def main():
    classes = config.get_classes()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading test sets...")
    ax, ay = collect_numpy(arduino_ds("test", classes))   # arducam (headline)
    mx, my = collect_numpy(main_ds("test", classes))      # main (generalization)

    print("Loading models...")
    float_model = tf.keras.models.load_model(config.FINAL_MODEL)

    md = []
    summary = []

    md.append("# Houseplant Classifier - Results\n")
    md.append("Generated by `report_metrics.py`. Models and test sets are the "
              "already-trained artifacts; this script only evaluates and reports.\n")
    md.append("- **FLOAT** = `models/plant_model.keras` (big float model)\n"
              "- **INT8** = `models/plant_model_int8.tflite` (deployed, post-training "
              "quantization)\n"
              "- **arducam test** = held-out OV7675 frames; **main test** = held-out "
              "Kaggle + phone images.\n")
    md.append("> Note: the arducam test frames are correlated with the arducam "
              "training frames (captured in the same sessions), so arducam test "
              "accuracy is optimistic; the live on-device demo is the true measure.\n")

    # placeholder for the summary table (filled after evaluation)
    summary_anchor = len(md)
    md.append("")  # will be replaced

    print("Evaluating FLOAT (keras)...")
    evaluate("FLOAT", lambda x: predict_keras(float_model, x), ax, ay, classes,
             "arducam", md, summary)
    evaluate("FLOAT", lambda x: predict_keras(float_model, x), mx, my, classes,
             "main", md, summary)

    print("Evaluating INT8 (tflite, deployed)...")
    evaluate("INT8", lambda x: predict_tflite(config.INT8_MODEL, x), ax, ay, classes,
             "arducam", md, summary)
    evaluate("INT8", lambda x: predict_tflite(config.INT8_MODEL, x), mx, my, classes,
             "main", md, summary)

    # Build the summary table now that we have every (model, set) row.
    sm = ["## Summary\n",
          "| Model | Test set | Accuracy | Macro-F1 | Weighted-F1 | Size |",
          "|---|---|---|---|---|---|"]
    size_kb = {
        "FLOAT": config.FLOAT_MODEL.stat().st_size / 1024 if config.FLOAT_MODEL.exists() else None,
        "INT8": config.INT8_MODEL.stat().st_size / 1024 if config.INT8_MODEL.exists() else None,
    }
    for name, tag, acc, mf, wf in summary:
        sz = f"{size_kb[name]:.0f} KB" if size_kb.get(name) else "-"
        sm.append(f"| {name} | {tag} | {acc:.4f} | {mf:.3f} | {wf:.3f} | {sz} |")
    sm.append("\nSize is the on-disk `.tflite` file size. The deployment-resource "
              "table below adds latency and on-device memory.\n")
    md[summary_anchor] = "\n".join(sm)

    # ---- Deployment-resource table (rubric Section 6 / 7) ----
    print("Measuring host latency...")
    lat_float = tflite_mean_latency_ms(config.FLOAT_MODEL, ax) \
        if config.FLOAT_MODEL.exists() else None
    lat_int8 = tflite_mean_latency_ms(config.INT8_MODEL, ax) \
        if config.INT8_MODEL.exists() else None
    int8_kb = config.INT8_MODEL.stat().st_size / 1024 if config.INT8_MODEL.exists() else None
    float_kb = config.FLOAT_MODEL.stat().st_size / 1024 if config.FLOAT_MODEL.exists() else None
    used_kb = config.ARENA_USED_BYTES / 1024
    alloc_kb = config.ARENA_ALLOC_BYTES / 1024

    dep = ["## Deployment resources\n",
           f"Target hardware: **{config.TARGET_HARDWARE}**. The INT8 model is the "
           f"one flashed to the device; the float model is shown for comparison only.\n",
           "| Model | Flash (model) | Tensor arena (used / reserved) | Latency (host)* | Output |",
           "|---|---|---|---|---|"]
    dep.append(
        f"| INT8 (deployed) | {int8_kb:.0f} KB | "
        f"{used_kb:.1f} KB / {alloc_kb:.0f} KB | "
        f"{lat_int8:.2f} ms | Serial dashboard |"
        if int8_kb is not None and lat_int8 is not None else
        "| INT8 (deployed) | - | - | - | Serial dashboard |")
    dep.append(
        f"| Float (reference) | {float_kb:.0f} KB | n/a (not deployed) | "
        f"{lat_float:.2f} ms | - |"
        if float_kb is not None and lat_float is not None else
        "| Float (reference) | - | - | - | - |")
    dep.append(
        "\n*Latency is a HOST estimate (mean single-frame `Invoke` over the arducam "
        "test set) for relative comparison; true on-device latency differs. "
        f"Tensor-arena 'used' is `arena_used_bytes()` measured on the "
        f"{config.TARGET_HARDWARE} at boot; 'reserved' is `kTensorArenaSize` in "
        "`plant_infer.ino`. The arena is the dominant RAM cost on the 256 KB device.\n")
    md.insert(summary_anchor + 1, "\n".join(dep))

    if not HAVE_MPL:
        md.append("_matplotlib was not available, so confusion-matrix PNGs were "
                  "skipped; the text confusion tables above are still complete._\n")

    out = RESULTS_DIR / "RESULTS.md"
    out.write_text("\n".join(md))
    print(f"\nWrote {out}")
    pngs = sorted(RESULTS_DIR.glob("cm_*.png"))
    if pngs:
        print(f"Wrote {len(pngs)} confusion-matrix PNG(s) to {RESULTS_DIR}/")
    print(f"sklearn: {'yes' if HAVE_SKLEARN else 'no (NumPy fallback)'}  |  "
          f"matplotlib: {'yes' if HAVE_MPL else 'no (PNGs skipped)'}")


if __name__ == "__main__":
    main()
