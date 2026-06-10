/*
  plant_infer.ino  -  on-device houseplant classifier (OV7675 + TFLM)

  Pipeline each frame:
    OV7675 QCIF RGB565  ->  center-crop square  ->  downsample to 96x96
      ->  quantize to int8  ->  MobileNetV1 (INT8) inference  ->  6 softmax scores
      ->  5-frame PEAK-HOLD + confidence THRESHOLD  ->  STAT/IMG over serial

  THREADED TX (mbed RTOS) for a smooth UI:
    - Main thread  (producer): readFrame -> preprocess -> Invoke -> peak-hold,
      then hands the result + a downsampled preview to the TX thread.
    - TX thread    (consumer): streams STAT + IMG over serial.
    Two semaphores (frame_ready / tx_done) pipeline them: the TX of frame N
    overlaps the capture + inference of frame N+1, so the ~18 KB serial write no
    longer blocks compute. The handshake also self-throttles (main waits if TX
    falls behind), and only the TX thread ever writes to Serial -> no races.

  Serial protocol to inference_pipeline/dashboard.py:
    STAT <idx> <conf> <c0..c5>\n            every inference (idx = peak-hold, -1 = uncertain)
    IMG 96 96\n 0x55AA55AA + 96*96*2 RGB565  the model's-eye preview (every SEND_IMG_EVERY)

  Setup: copy models/model_int8.h into this folder (next to plant_infer.ino).
  Folder must be named plant_infer to open in the Arduino IDE.
*/

#include "mbed.h"
#include <TensorFlowLite.h>
#include <TinyMLShield.h>
#include "model_int8.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ---- configuration ----
#define CAM_W       176      // QCIF (QQVGA gave invalid frames on the OV7675 driver)
#define CAM_H       144
#define IN_DIM      96       // model input (must match config.IMG_SIZE)
#define N_CLASSES   6
#define WINDOW      5        // peak-hold window
#define THRESHOLD   0.60f    // below this -> "No houseplant detected" (idx = -1)
#define SEND_IMG_EVERY 1     // stream the 96x96 model-view preview every N inferences
#define IMG_BYTES   (IN_DIM * IN_DIM * 2)

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// MobileNetV1 96x96 measured ~103 KB; 112 KB leaves margin while freeing RAM for
// the TX thread's stack + the preview buffer below.
constexpr int kTensorArenaSize = 112 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// QCIF RGB565 camera frame buffer (owned by the main thread).
byte frame[CAM_W * CAM_H * 2];

// 5-frame peak-hold ring buffer (main thread).
int   winLabel[WINDOW];
float winConf[WINDOW];
int   winPos = 0, winCount = 0;

// ---- producer/consumer handoff between main and the TX thread ----
struct Payload {
  int      idx;                 // peak-hold class (-1 = uncertain)
  float    peak;                // peak-hold confidence
  float    conf[N_CLASSES];     // current-frame class scores
  bool     has_img;             // whether to send the preview this round
  uint8_t  img[IMG_BYTES];      // downsampled RGB565 preview (model's-eye view)
};
Payload shared;

rtos::Thread    tx_thread(osPriorityNormal, 4096);
rtos::Semaphore frame_ready(0, 1);   // main -> TX: a payload is ready
rtos::Semaphore tx_done(1, 1);       // TX -> main: shared buffer is free (starts free)
}  // namespace

// Decode one big-endian RGB565 pixel from the camera frame -> 8-bit r,g,b.
static inline void rgb565_at(int x, int y, int& r, int& g, int& b) {
  int i = (y * CAM_W + x) * 2;
  uint16_t px = ((uint16_t)frame[i] << 8) | frame[i + 1];
  r = ((px >> 11) & 0x1F) << 3;
  g = ((px >> 5) & 0x3F) << 2;
  b = (px & 0x1F) << 3;
}

// Center-crop the frame to square, downsample to IN_DIM, write the int8 input tensor.
static void preprocess() {
  const float s = input->params.scale;
  const int   zp = input->params.zero_point;
  const int   crop = CAM_H;
  const int   x0 = (CAM_W - crop) / 2;
  int8_t* d = input->data.int8;
  int k = 0;
  for (int oy = 0; oy < IN_DIM; oy++) {
    int sy = oy * crop / IN_DIM;
    for (int ox = 0; ox < IN_DIM; ox++) {
      int sx = x0 + ox * crop / IN_DIM;
      int r, g, b;
      rgb565_at(sx, sy, r, g, b);
      int qr = lroundf(r / s) + zp; qr = qr < -128 ? -128 : (qr > 127 ? 127 : qr);
      int qg = lroundf(g / s) + zp; qg = qg < -128 ? -128 : (qg > 127 ? 127 : qg);
      int qb = lroundf(b / s) + zp; qb = qb < -128 ? -128 : (qb > 127 ? 127 : qb);
      d[k++] = (int8_t)qr; d[k++] = (int8_t)qg; d[k++] = (int8_t)qb;
    }
  }
}

// Downsample the raw frame[] to the model's-eye view (big-endian RGB565) into a
// buffer for the TX thread. Reads frame[] (valid), not the int8 tensor (clobbered
// by Invoke). Runs in the main thread while TX is idle, so frame[] isn't shared.
static void fill_preview(uint8_t* out) {
  const int crop = CAM_H;
  const int x0 = (CAM_W - crop) / 2;
  int k = 0;
  for (int oy = 0; oy < IN_DIM; oy++) {
    int sy = oy * crop / IN_DIM;
    for (int ox = 0; ox < IN_DIM; ox++) {
      int sx = x0 + ox * crop / IN_DIM;
      int r, g, b;
      rgb565_at(sx, sy, r, g, b);
      uint16_t px = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
      out[k++] = (uint8_t)(px >> 8);
      out[k++] = (uint8_t)(px & 0xFF);
    }
  }
}

// TX thread: the ONLY writer of Serial during normal operation. Blocks until the
// main thread signals a ready payload, streams it, then frees the buffer.
static void tx_loop() {
  while (true) {
    frame_ready.acquire();

    Serial.print("STAT ");
    Serial.print(shared.idx);
    Serial.print(' ');
    Serial.print(shared.peak, 3);
    for (int i = 0; i < N_CLASSES; i++) { Serial.print(' '); Serial.print(shared.conf[i], 3); }
    Serial.println();

    if (shared.has_img) {
      Serial.print("IMG "); Serial.print(IN_DIM); Serial.print(' '); Serial.println(IN_DIM);
      Serial.write(0x55); Serial.write(0xAA); Serial.write(0x55); Serial.write(0xAA);  // sync magic
      Serial.write(shared.img, IMG_BYTES);
    }

    tx_done.release();
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 4000);
  delay(500);
  Serial.println("boot: serial up");

  if (!Camera.begin(QCIF, RGB565, 1, OV7675)) {
    Serial.println("ERR: camera init failed");
    while (1);
  }
  Serial.println("boot: camera ok");

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(plant_model_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERR: model schema mismatch");
    while (1);
  }

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
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERR: AllocateTensors failed (raise arena or lower IMG_SIZE)");
    while (1);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("boot: arena used ");
  Serial.println(interpreter->arena_used_bytes());
  Serial.println("boot: ready");   // if you see this + STAT lines, RAM fits

  // Start the serial-TX thread. From here on, only this thread writes Serial.
  tx_thread.start(tx_loop);
}

void loop() {
  Camera.readFrame(frame);
  preprocess();

  if (interpreter->Invoke() != kTfLiteOk) {
    return;   // drop the frame; avoid writing Serial here (TX thread owns it)
  }

  // Dequantize the 6 outputs; find this frame's best class.
  const float os = output->params.scale;
  const int   ozp = output->params.zero_point;
  float conf[N_CLASSES];
  int best = 0;
  for (int i = 0; i < N_CLASSES; i++) {
    conf[i] = (output->data.int8[i] - ozp) * os;
    if (conf[i] > conf[best]) best = i;
  }

  // Push into the peak-hold window.
  winLabel[winPos] = best;
  winConf[winPos] = conf[best];
  winPos = (winPos + 1) % WINDOW;
  if (winCount < WINDOW) winCount++;

  // Peak-hold: the most confident frame in the window.
  int pk = 0;
  for (int i = 1; i < winCount; i++)
    if (winConf[i] > winConf[pk]) pk = i;
  int outIdx = (winConf[pk] >= THRESHOLD) ? winLabel[pk] : -1;
  float outConf = winConf[pk];

  // ---- hand off to the TX thread (producer side) ----
  // Wait until the TX thread has finished the previous payload, so we never
  // overwrite shared mid-send. While TX was sending, this thread already did the
  // next readFrame + Invoke above -> that's the overlap.
  tx_done.acquire();
  shared.idx = outIdx;
  shared.peak = outConf;
  for (int i = 0; i < N_CLASSES; i++) shared.conf[i] = conf[i];
  static int frameCount = 0;
  shared.has_img = ((frameCount++ % SEND_IMG_EVERY) == 0);
  if (shared.has_img) fill_preview(shared.img);   // frame[] is ours right now
  frame_ready.release();
}
