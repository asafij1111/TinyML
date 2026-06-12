# Houseplant Classifier (TinyML, Arduino Nano 33 BLE Sense)

A six-class houseplant species classifier that runs entirely on an Arduino Nano 33
BLE Sense (Nordic nRF52840) with an Arducam OV7675 camera. The model classifies a
live camera frame into one of six species on-device, with no network or cloud
inference, and streams the result to a local dashboard over USB serial.

Classes: Aloe Vera, Asparagus Fern, Dumb Cane, Jade Plant, Money Tree, Snake Plant.

## How it works

A MobileNetV1 backbone (alpha = 0.25, ImageNet-pretrained, 96x96 RGB input) is
trained in three stages and then quantized to full-integer INT8 for the
microcontroller:

1. Head training on a general dataset (Kaggle photos + phone photos).
2. Backbone fine-tuning on the same general dataset.
3. Domain specialization on images captured through the OV7675 camera itself, so
   the model adapts to the real deployment sensor and environment. This stage uses
   layer freezing, a low learning rate, early stopping, and rehearsal (mixing in a
   fraction of general data) to specialize without forgetting.

The trained model is converted to INT8 TFLite by post-training quantization
(303 KB, ~100 KB tensor arena) and runs on-device under TensorFlow Lite for
Microcontrollers. Each frame is captured, center-cropped and downsampled to 96x96,
quantized, and classified; a 5-frame peak-hold plus a confidence threshold smooth
the output. Results and a preview of the model's input are sent over serial to a
matplotlib dashboard.

## Repository layout

```
Final_Project/
  config.py                 Central config: paths, image size, splits, LRs, seed,
                            and the on-device deployment measurements.
  prepare_data.py           Build the general train/val/test split (Kaggle + phone).
  prepare_arduino.py        Build the arducam (camera) train/val/test split.
  train.py                  Three-stage training -> models/plant_model.keras.
  quantize.py               Float + INT8 TFLite conversion and a 4-metric table.
  report_metrics.py         Generate results/RESULTS.md + confusion-matrix PNGs.
  models/                   Trained model, TFLite files, C header, labels.
  results/                  Generated metrics report and figures.
  data_collection_pipeline/
    plant_capture/          Arduino sketch: capture one OV7675 frame on request.
    capture_serial.py       Host script: save captured frames as a labeled dataset.
  inference_pipeline/
    plant_infer/            Arduino sketch: on-device inference + serial protocol.
    arena_probe/            Arduino sketch: measure the tensor-arena size.
    dashboard.py            Host dashboard: live prediction, confidence, preview.
  presentation/             Slide generators and the exported report PDF.
```

The image datasets (`house_plant_species/`, `data_split/`, `arduino_split/`,
`collected_images/`, `collected_arduino/`) are not committed; see Data below.

## Requirements

- Python 3.11 with TensorFlow 2.14, NumPy, Pillow, Matplotlib, scikit-learn, pySerial.
- Arduino IDE with the Arduino Mbed OS Nano core, the Arduino_TensorFlowLite
  library, and the Harvard TinyMLx camera support (TinyMLShield / Arduino_OV767X).
- Hardware: Arduino Nano 33 BLE Sense + Arducam OV7675 camera module.

## Data

The general dataset is the public "House Plant Species" image dataset from Kaggle
(search Kaggle for "House Plant Species"). Download it and place the six class
folders under `house_plant_species/`. The phone photos and the OV7675 camera
captures are the author's own; the camera captures can be reproduced with the
data-collection pipeline below. The source folders are treated as read-only: the
prepare scripts only ever copy out of them.

## Reproduce

All commands run from the `Final_Project/` directory.

1. Build the splits:

   ```
   python prepare_data.py        # general dataset -> data_split/
   python prepare_arduino.py     # camera dataset  -> arduino_split/
   ```

2. Train (three stages) and quantize:

   ```
   python train.py               # -> models/plant_model.keras, labels.txt
   python quantize.py            # -> models/plant_model_int8.tflite (+ float)
   ```

3. Generate the results report (optional but recommended):

   ```
   python report_metrics.py      # -> results/RESULTS.md + confusion-matrix PNGs
   ```

4. Embed the model for firmware (quantize.py prints this command):

   ```
   xxd -i models/plant_model_int8.tflite > models/model_int8.h
   ```

   Then mark the array `alignas(16) const` so it is placed in flash, and copy
   `model_int8.h` next to `inference_pipeline/plant_infer/plant_infer.ino`.

5. Flash and run:

   - Open `inference_pipeline/plant_infer/plant_infer.ino` in the Arduino IDE and
     upload it to the Nano 33 BLE Sense (double-tap RESET to enter the bootloader
     if the board is not detected).
   - Run the dashboard on the host, pointing it at the board's serial port:

     ```
     python inference_pipeline/dashboard.py
     ```

To collect your own camera data for the domain-specialization stage, upload
`data_collection_pipeline/plant_capture/plant_capture.ino` and run
`data_collection_pipeline/capture_serial.py`, which saves one labeled frame per
request into `collected_arduino/`.

## Notes

- All key hyperparameters (image size, split ratios, learning rates, epochs,
  rehearsal weight, random seed = 42) live in `config.py`, so a run is fully
  determined by that file plus the data.
- The on-device tensor-arena measurement in the deployment table is read from the
  board's boot serial print and recorded in `config.py`.
- MobileNetV1 is used (not V2/V3) because V2/V3 inverted-residual blocks expand
  activations to roughly 293 KB at 96x96, which exceeds the device's 256 KB RAM;
  MobileNetV1 fits in about 100 KB.
