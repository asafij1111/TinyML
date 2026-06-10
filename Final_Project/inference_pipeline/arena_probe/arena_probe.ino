/*
  arena_probe.ino  -  measure the TFLM tensor-arena size for the plant model.

  This is the #1 firmware de-risk: it loads the INT8 model, builds the op
  resolver, allocates tensors, and prints how many bytes the arena actually
  needs. That number tells us whether the model + a camera frame buffer fit in
  the Nano 33 BLE Sense's 256 KB RAM, and at what capture resolution.

  No camera here on purpose -- we isolate the RAM question.

  Setup:
    1. Generate the model header (from the project's models/ dir):
         cd models && xxd -i plant_model_int8.tflite > model_int8.h
       -> defines  plant_model_int8_tflite[]  and  plant_model_int8_tflite_len
    2. Copy model_int8.h into THIS sketch folder (next to arena_probe.ino).
    3. Open in Arduino IDE (folder must be named arena_probe), select the
       Nano 33 BLE, upload, open Serial Monitor @ 115200.

  Read the "Arena used (bytes)" line. If you instead see "AllocateTensors
  FAILED", raise kArenaSize below and re-upload.
*/

#include <TensorFlowLite.h>
#include "model_int8.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// Generous ceiling for the probe (no camera buffer competing yet). If
// AllocateTensors fails, this was too small -- raise it. If the sketch won't
// even start, it's too big for RAM -- lower it.
constexpr int kArenaSize = 180 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];
}  // namespace

void setup() {
  Serial.begin(115200);
  // Don't block forever on Serial (some hosts never assert DTR); give the
  // monitor a few seconds to attach, then proceed regardless.
  while (!Serial && millis() < 4000);
  delay(800);
  Serial.println("boot: serial up");           // <-- proves the board is alive

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  Serial.println("boot: loading model...");
  model = tflite::GetModel(plant_model_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERR: model schema ");
    Serial.print(model->version());
    Serial.print(" != ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1);
  }

  // The 8 ops in plant_model_int8.tflite (from quantize.py inspection).
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddAdd();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddMul();
  resolver.AddPad();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize, error_reporter);
  interpreter = &static_interpreter;

  Serial.println("boot: allocating tensors...");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors FAILED - raise kArenaSize and re-upload.");
    while (1);
  }

  Serial.println("==== arena probe ====");
  Serial.print("Arena used (bytes): ");
  Serial.println(interpreter->arena_used_bytes());
  Serial.print("Arena ceiling     : ");
  Serial.println(kArenaSize);

  TfLiteTensor* in = interpreter->input(0);
  TfLiteTensor* out = interpreter->output(0);
  Serial.print("input dims        : ");
  for (int i = 0; i < in->dims->size; i++) { Serial.print(in->dims->data[i]); Serial.print(' '); }
  Serial.println();
  Serial.print("input scale/zp    : ");
  Serial.print(in->params.scale, 6); Serial.print(' '); Serial.println(in->params.zero_point);
  Serial.print("output scale/zp   : ");
  Serial.print(out->params.scale, 6); Serial.print(' '); Serial.println(out->params.zero_point);
  Serial.println("Done. Use 'Arena used' to budget the camera frame buffer.");
}

void loop() {}
