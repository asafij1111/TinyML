/*
  plant_capture.ino  -  OV7675 dataset capture for the Plant Detector project

  Streams camera frames to the Mac over USB serial so capture_serial.py can save
  them as images. Built on the course TinyMLShield / OV767X library (same API the
  prof used in Lab5's CameraCaptureRawBytes), with two additions for reliable
  bulk capture:

    1. Request/response: the Mac sends 'c' to request ONE frame. This lets the
       Python side pace capture (e.g. ~3 fps) while you move the camera around
       the plant, instead of the board free-running faster than you can move.

    2. Sync header before every frame: MAGIC (0x55AA55AA) + width + height (each
       big-endian uint16). The Python side resyncs on MAGIC each frame, so a
       dropped byte or startup garbage can't shear the whole capture session.

  Pixel format is RGB565, big-endian per pixel (matches the prof's visualizer).

  NOTE for Arduino IDE: to compile, this file must live in a folder named
  "plant_capture" (folder name == sketch name). It is kept here for now; move the
  whole folder into your Arduino sketches when you're ready to flash.
*/

#include <TinyMLShield.h>

// QVGA = 320x240, the highest resolution that fits this board's RAM: a QVGA
// RGB565 frame buffer is 320*240*2 = 153600 bytes (VGA 640x480 would need 614KB,
// far over the 256KB total). We still downsample to 96x96 at train time, but
// capturing at QVGA gives a sharper source image. Drop to QCIF (176x144) for
// faster transfer.
#define CAM_RES   QVGA
#define CAM_FMT   RGB565

byte frame[320 * 240 * 2];   // QVGA RGB565 = 153600 bytes (fits 256KB RAM)
int  bytesPerFrame;

void writeU16(uint16_t v) {  // big-endian, matches Python '>u2' decode
  Serial.write((uint8_t)(v >> 8));
  Serial.write((uint8_t)(v & 0xFF));
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!Camera.begin(CAM_RES, CAM_FMT, 1, OV7675)) {
    Serial.println("ERR: camera init failed");
    while (1);
  }
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
}

void loop() {
  if (Serial.available() && Serial.read() == 'c') {
    Camera.readFrame(frame);

    // ---- sync header ----
    Serial.write(0x55); Serial.write(0xAA);
    Serial.write(0x55); Serial.write(0xAA);
    writeU16(Camera.width());
    writeU16(Camera.height());

    // ---- raw RGB565 frame ----
    Serial.write(frame, bytesPerFrame);
  }
}
