// Standalone References slide (matches the deck style).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "References";

const INK = "1F2937", MUTE = "6B7280";
const GREEN = "2C5F2D", LINEC = "E5E7EB";
const HEAD = "Helvetica Neue", BODY = "Helvetica Neue";
const W = 10, M = 0.65;

const s = pres.addSlide();
s.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.5, w: 0.14, h: 0.14, fill: { color: GREEN }, line: { type: "none" } });
s.addText("APPENDIX", { x: M + 0.26, y: 0.43, w: 8, h: 0.3, margin: 0, fontFace: BODY, fontSize: 11, bold: true, color: GREEN, charSpacing: 3 });
s.addText("References", { x: M, y: 0.75, w: W - 2 * M, h: 0.6, margin: 0, fontFace: HEAD, fontSize: 28, bold: true, color: INK });

function group(x, y, w, title, items) {
  s.addText(title, { x, y, w, h: 0.28, margin: 0, fontFace: HEAD, fontSize: 13, bold: true, color: GREEN });
  s.addText(items.map((t, i) => ({ text: t, options: {
    bullet: { indent: 12 }, breakLine: i < items.length - 1,
    fontFace: BODY, fontSize: 10.5, color: INK, paraSpaceAfter: 6 } })),
    { x, y: y + 0.3, w, h: 2.0, margin: 0, valign: "top" });
}

// ----- left column -----
group(M, 1.5, 4.35, "Dataset", [
  "“House Plant Species” image dataset — Kaggle.",
]);
group(M, 2.35, 4.35, "Models & papers", [
  "Howard et al., “MobileNets: Efficient CNNs for Mobile Vision Applications,” arXiv:1704.04861, 2017.",
  "Sandler et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” CVPR 2018 (arXiv:1801.04381).",
  "ImageNet — pretrained backbone weights.",
]);

// ----- right column -----
const RX = 5.3, RW = 4.05;
group(RX, 1.5, RW, "Frameworks & tools", [
  "TensorFlow / Keras — model training.",
  "TensorFlow Lite + TFLite for Microcontrollers — INT8 quantization & on-device inference.",
  "Arduino IDE + Arduino Mbed OS Nano core.",
]);
group(RX, 3.0, RW, "Hardware", [
  "Arduino Nano 33 BLE Sense (Nordic nRF52840).",
  "Arducam OV7675 0.3 MP camera module.",
]);
group(RX, 4.1, RW, "Libraries & reused code", [
  "Arduino_TensorFlowLite library.",
  "Harvard TinyMLx — TinyMLShield / Arduino_OV767X; camera + TFLM examples adapted from EE446 Lab 5 (CameraCaptureRawBytes, person_detection).",
  "Python: NumPy, Pillow, pySerial, Matplotlib.",
]);

pres.writeFile({ fileName: "references_slide.pptx" }).then(f => console.log("wrote", f));
