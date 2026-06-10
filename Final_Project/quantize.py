"""Convert the trained model to TFLite (float + INT8) and report the 4 metrics.

Produces, from models/plant_model.keras:
  - models/plant_model_float.tflite   (float32 reference)
  - models/plant_model_int8.tflite    (full-integer, the DEPLOYED model)
and prints a comparison table of the four required metrics:
  accuracy (arducam test) | model size | latency | peak RAM
for the float and INT8 variants.

INT8 calibration uses a representative dataset drawn from the arducam TRAIN
split, so the quantization ranges match the deployment camera. Input/output are
kept INT8 to match the MCU (TFLM). Latency/RAM here are HOST estimates for
relative comparison -- the true on-device latency and tensor-arena size come
from the Arduino sketch after AllocateTensors().

Run:  python quantize.py   (after train.py)
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TF C++ info/warnings
import warnings
warnings.filterwarnings("ignore")
import time

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import config
from train import arduino_ds, main_ds


def collect_numpy(ds, limit=None):
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(int(y))
        if limit and len(xs) >= limit:
            break
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def representative_dataset(classes):
    """Yield single 0..255 float frames from the arducam train split."""
    ds = arduino_ds("train", classes)
    def gen():
        for i, (x, _) in enumerate(ds):
            if i >= config.REP_SAMPLES:
                break
            yield [tf.expand_dims(x, 0)]
    return gen


def convert_float(model):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    return conv.convert()


def convert_int8(model, classes):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(classes)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_eval(tflite_bytes, x, y):
    """Accuracy + mean latency (ms) of a TFLite model on (x, y)."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

    def prep(img):
        if inp["dtype"] == np.int8:
            scale, zp = inp["quantization"]
            q = np.round(img / scale) + zp
            return np.clip(q, -128, 127).astype(np.int8)
        return img.astype(np.float32)

    correct, times = 0, []
    for img, label in zip(x, y):
        interp.set_tensor(inp["index"], prep(img)[None])
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
        pred = int(np.argmax(interp.get_tensor(out["index"])[0]))
        correct += (pred == label)
    # loose host RAM proxy: sum of all tensor buffer sizes (the device arena is
    # smaller/measured separately, but this orders the variants correctly)
    ram = sum(int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
              for d in interp.get_tensor_details() if len(d["shape"]))
    return correct / len(y), float(np.mean(times)), ram


def main():
    classes = config.get_classes()
    model = tf.keras.models.load_model(config.FINAL_MODEL)

    print("Loading test sets...")
    ax, ay = collect_numpy(arduino_ds("test", classes))      # arducam (headline)
    mx, my = collect_numpy(main_ds("test", classes))         # main (generalization)

    print("Converting float TFLite...")
    float_tfl = convert_float(model)
    config.FLOAT_MODEL.write_bytes(float_tfl)
    print("Converting INT8 TFLite (arducam-calibrated)...")
    int8_tfl = convert_int8(model, classes)
    config.INT8_MODEL.write_bytes(int8_tfl)

    rows = []
    for name, blob in [("float", float_tfl), ("INT8 (deployed)", int8_tfl)]:
        acc_ard, lat, ram = tflite_eval(blob, ax, ay)
        acc_main, _, _ = tflite_eval(blob, mx, my)
        rows.append((name, acc_ard, acc_main, len(blob), lat, ram))

    print("\n=================================== METRICS ===================================")
    print(f"{'variant':<16} {'arducam acc':>11} {'main acc':>9} "
          f"{'size (KB)':>10} {'latency ms':>11} {'RAM~KB (host)':>14}")
    print("-" * 78)
    for name, a_ard, a_main, size, lat, ram in rows:
        print(f"{name:<16} {a_ard:>11.4f} {a_main:>9.4f} "
              f"{size/1024:>10.1f} {lat:>11.2f} {ram/1024:>14.1f}")
    print("-" * 78)
    print("Notes: latency/RAM are HOST estimates for relative comparison; the true")
    print("on-device latency + tensor-arena size come from the Arduino sketch.")

    # input quantization params (firmware needs these to feed camera pixels)
    interp = tf.lite.Interpreter(model_content=int8_tfl)
    interp.allocate_tensors()
    iscale, izp = interp.get_input_details()[0]["quantization"]
    print(f"\nINT8 input quant: scale={iscale:.6f} zero_point={izp}")
    print(f"  firmware: int8_pixel = round(pixel / {iscale:.6f}) + ({izp})")

    print(f"\nEmbed for firmware:\n  xxd -i {config.INT8_MODEL} > {config.MODEL_HEADER}")


if __name__ == "__main__":
    main()
