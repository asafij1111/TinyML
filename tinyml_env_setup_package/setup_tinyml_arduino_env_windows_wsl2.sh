#!/usr/bin/env bash
set -Eeuo pipefail

# Stable TinyML Python environment for Windows via WSL2 Ubuntu
# Run this INSIDE the Ubuntu terminal in WSL2, not in PowerShell or CMD.
#
# Usage:
#   bash setup_tinyml_arduino_env_windows_wsl2.sh
#   bash setup_tinyml_arduino_env_windows_wsl2.sh tinyml-arduino ~/ai/projects
#
# Optional environment variables:
#   RUN_SMOKE_TEST=0         Skip the smoke test at the end
#   REINSTALL=1              Delete and recreate the environment if it exists
#   INSTALL_ARDUINO_TFLM=1   Clone the Arduino TensorFlow Lite Micro examples library

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  printf '\nERROR: This script is intended for WSL2 Ubuntu on Windows.\n' >&2
  printf 'Open your Ubuntu terminal in WSL2 and run it there.\n' >&2
  exit 1
fi

ENV_NAME="${1:-tinyml-arduino}"
BASE_DIR="${2:-$HOME/ai/projects}"
ENV_DIR="$BASE_DIR/$ENV_NAME"
KERNEL_NAME="$ENV_NAME"
DISPLAY_NAME="Python ($ENV_NAME)"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
REINSTALL="${REINSTALL:-0}"
INSTALL_ARDUINO_TFLM="${INSTALL_ARDUINO_TFLM:-0}"

TF_VERSION="2.14.1"
KERAS_VERSION="2.14.0"
TFMOT_VERSION="0.8.0"
NUMPY_SPEC="numpy<2"

msg() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

find_python311() {
  if command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
    return 0
  fi
  return 1
}

ensure_python311() {
  if PY311="$(find_python311)"; then
    echo "$PY311"
    return 0
  fi

  msg "python3.11 not found. Installing required Ubuntu packages..."
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3-pip build-essential git
  if PY311="$(find_python311)"; then
    echo "$PY311"
    return 0
  fi

  fail "Could not locate python3.11 after installation."
}

msg "Preparing stable TinyML environment on Windows via WSL2: $ENV_NAME"
mkdir -p "$BASE_DIR"

PY311="$(ensure_python311)"
msg "Using Python interpreter: $PY311"
"$PY311" --version

if [[ -d "$ENV_DIR" && "$REINSTALL" == "1" ]]; then
  msg "Removing existing environment because REINSTALL=1: $ENV_DIR"
  rm -rf "$ENV_DIR"
fi

if [[ ! -d "$ENV_DIR" ]]; then
  msg "Creating virtual environment at: $ENV_DIR"
  "$PY311" -m venv "$ENV_DIR"
else
  msg "Environment already exists: $ENV_DIR"
fi

# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

msg "Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

msg "Installing pinned TinyML packages"
python -m pip install \
  "tensorflow==$TF_VERSION" \
  "keras==$KERAS_VERSION" \
  "tensorflow-model-optimization==$TFMOT_VERSION" \
  "$NUMPY_SPEC" \
  pandas \
  scipy \
  scikit-learn \
  matplotlib \
  seaborn \
  jupyterlab \
  ipykernel \
  notebook \
  pyserial \
  tqdm \
  tabulate

msg "Installing Jupyter kernel: $DISPLAY_NAME"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

if [[ "$INSTALL_ARDUINO_TFLM" == "1" ]]; then
  ARDUINO_LIB_DIR="$HOME/Arduino/libraries"
  TARGET_DIR="$ARDUINO_LIB_DIR/Arduino_TensorFlowLite"
  mkdir -p "$ARDUINO_LIB_DIR"
  if [[ ! -d "$TARGET_DIR" ]]; then
    if command -v git >/dev/null 2>&1; then
      msg "Cloning TensorFlow Lite Micro Arduino examples library"
      git clone https://github.com/tensorflow/tflite-micro-arduino-examples "$TARGET_DIR"
    else
      msg "Skipping Arduino TFLM library clone because git is not installed"
    fi
  else
    msg "Arduino TFLM library already exists: $TARGET_DIR"
  fi
fi

SMOKE_TEST_PATH="$ENV_DIR/tinyml_env_smoke_test.py"
cat > "$SMOKE_TEST_PATH" <<'PY'
import os
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

print("TensorFlow:", tf.__version__)
print("TFMOT:", tfmot.__version__)

np.random.seed(42)
tf.random.set_seed(42)

num_samples = 800
input_dim = 20
num_classes = 4

X = np.random.randn(num_samples, input_dim).astype(np.float32)
y = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)

X_train, X_test = X[:600], X[600:]
y_train, y_test = y[:600], y[600:]

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

baseline_model = build_model()
baseline_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
baseline_model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
loss, acc = baseline_model.evaluate(X_test, y_test, verbose=0)
print(f"Baseline accuracy: {acc:.4f}")

converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
tflite_fp32 = converter.convert()
print("FP32 TFLite bytes:", len(tflite_fp32))

converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_drq = converter.convert()
print("Dynamic range quantized bytes:", len(tflite_drq))

def representative_data_gen():
    for i in range(100):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(baseline_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_int8 = converter.convert()
print("Full INT8 quantized bytes:", len(tflite_int8))

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=int(np.ceil(len(X_train) / 32)) * 3,
    )
}
pruned_model = prune_low_magnitude(build_model(), **pruning_params)
pruned_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
pruned_model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=32,
    verbose=0,
    callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
)
pruned_model_stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model_stripped)
tflite_pruned = converter.convert()
print("Pruned model TFLite bytes:", len(tflite_pruned))

qat_model = tfmot.quantization.keras.quantize_model(build_model())
qat_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
qat_model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_qat_int8 = converter.convert()
print("QAT INT8 TFLite bytes:", len(tflite_qat_int8))

teacher = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
teacher.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
teacher.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
soft_labels = teacher.predict(X_train, verbose=0)

student_distill = build_model()
student_distill.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['accuracy'])
student_distill.fit(X_train, soft_labels, epochs=3, batch_size=32, verbose=0)
converter = tf.lite.TFLiteConverter.from_keras_model(student_distill)
tflite_distilled = converter.convert()
print("Distilled student TFLite bytes:", len(tflite_distilled))

output_path = os.path.join(tempfile.gettempdir(), "tinyml_model_int8.tflite")
with open(output_path, "wb") as f:
    f.write(tflite_int8)

header_path = os.path.join(tempfile.gettempdir(), "tinyml_model_int8.h")
with open(output_path, "rb") as f:
    model_bytes = f.read()

with open(header_path, "w") as f:
    f.write("const unsigned char tinyml_model_int8[] = {")
    for i, b in enumerate(model_bytes):
        if i % 12 == 0:
            f.write("\n  ")
        f.write(f"0x{b:02x}, ")
    f.write("\n};\n")
    f.write(f"const unsigned int tinyml_model_int8_len = {len(model_bytes)};\n")

print("C header exported to:", header_path)
print("ALL SMOKE TESTS PASSED")
PY

cat > "$ENV_DIR/launch_jupyter.sh" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
source "$ENV_DIR/bin/activate"
exec jupyter lab --no-browser --ip=0.0.0.0
EOF2
chmod +x "$ENV_DIR/launch_jupyter.sh"

if [[ "$RUN_SMOKE_TEST" == "1" ]]; then
  msg "Running TinyML smoke test"
  python "$SMOKE_TEST_PATH"
else
  msg "Skipping smoke test because RUN_SMOKE_TEST=0"
fi

cat <<EOF2

Setup complete.

Environment:
  $ENV_DIR

Activate:
  source "$ENV_DIR/bin/activate"

Deactivate:
  deactivate

Launch JupyterLab:
  "$ENV_DIR/launch_jupyter.sh"

Then open the localhost Jupyter link shown in the Ubuntu terminal from your Windows browser.

Run smoke test later:
  python "$SMOKE_TEST_PATH"

Jupyter kernel:
  $DISPLAY_NAME

EOF2
