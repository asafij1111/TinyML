// Plant Detector — EE446 TinyML final presentation deck (minimal style).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";          // 10" x 5.625"
pres.author = "Asaf Iron-Jobes";
pres.title = "Plant Detector — TinyML";

// ---- palette (minimal: charcoal on white, one forest-green accent) ----
const INK = "1F2937";
const MUTE = "6B7280";
const FAINT = "9CA3AF";
const GREEN = "2C5F2D";
const PALE = "F4F6F4";
const LINEC = "E5E7EB";
const WHITE = "FFFFFF";
const DARK = "16241A";          // deep green-charcoal for title/closing
const HEAD = "Helvetica Neue";  // falls back gracefully
const BODY = "Helvetica Neue";

const W = 10, H = 5.625, M = 0.65;

// kicker (small green section label) + big title, consistent across slides
function header(slide, kicker, title) {
  slide.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.55, w: 0.14, h: 0.14, fill: { color: GREEN }, line: { type: "none" } });
  slide.addText(kicker.toUpperCase(), { x: M + 0.26, y: 0.48, w: 8, h: 0.3, margin: 0,
    fontFace: BODY, fontSize: 11, bold: true, color: GREEN, charSpacing: 3 });
  slide.addText(title, { x: M, y: 0.82, w: W - 2 * M, h: 0.7, margin: 0,
    fontFace: HEAD, fontSize: 30, bold: true, color: INK });
}

// labeled placeholder box for the user's own screenshots / video
function placeholder(slide, x, y, w, h, label) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: PALE }, line: { color: LINEC, width: 1, dashType: "dash" } });
  slide.addText(label, { x, y, w, h, margin: 0, align: "center", valign: "middle",
    fontFace: BODY, fontSize: 12, italic: true, color: FAINT });
}

function bullets(slide, items, x, y, w, opt = {}) {
  const arr = items.map((t, i) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: true,
    fontFace: BODY, fontSize: opt.fs || 15, color: opt.color || INK, paraSpaceAfter: opt.gap || 9 } }));
  slide.addText(arr, { x, y, w, h: opt.h || 4, margin: 0, valign: "top" });
}

// ===================================================================== 1. TITLE
let s = pres.addSlide();
s.background = { color: DARK };
s.addShape(pres.shapes.RECTANGLE, { x: M, y: 1.7, w: 0.5, h: 0.14, fill: { color: GREEN }, line: { type: "none" } });
s.addText("Plant Detector", { x: M, y: 1.9, w: 9, h: 1.0, margin: 0, fontFace: HEAD, fontSize: 54, bold: true, color: WHITE });
s.addText("On-device houseplant classifier · TinyML", { x: M, y: 2.95, w: 9, h: 0.5, margin: 0, fontFace: BODY, fontSize: 20, color: "C7D2CC" });
s.addText([
  { text: "Asaf Iron-Jobes", options: { bold: true, color: WHITE } },
  { text: "   ·   EE446 TinyML — Final Project", options: { color: "9FB0A4" } },
], { x: M, y: 4.15, w: 9, h: 0.4, margin: 0, fontFace: BODY, fontSize: 14 });
s.addText("Arduino Nano 33 BLE Sense  ·  OV7675 camera  ·  MobileNetV1  ·  TensorFlow Lite Micro",
  { x: M, y: 4.55, w: 9, h: 0.4, margin: 0, fontFace: BODY, fontSize: 12, color: "7E9184" });

// ===================================================================== 2. OVERVIEW
s = pres.addSlide();
header(s, "Overview", "Classify 6 houseplants, entirely on the chip");
bullets(s, [
  "Live camera frame → species prediction + confidence, with no cloud or PC inference.",
  "Runs in 256 KB RAM / 1 MB flash on the Nano 33 BLE Sense (nRF52840, Cortex-M4).",
  "Streams the prediction + the model's-eye image to a desktop dashboard over USB serial.",
  "Specialized to the exact plants, camera, and table used in the demo.",
], M, 1.7, 5.4, { gap: 12 });
placeholder(s, 6.35, 1.7, 3.0, 3.2, "[ demo photo / video still ]");
s.addText("6 classes · ~0.82 on-device test accuracy (INT8) · ~100 KB tensor arena",
  { x: M, y: 5.0, w: 8.7, h: 0.3, margin: 0, fontFace: BODY, fontSize: 11, italic: true, color: MUTE });

// ===================================================================== 3. CLASSES
s = pres.addSlide();
header(s, "Target plants", "The six species");
const classes = [
  ["Aloe Vera", "succulent, spiky rosette"],
  ["Asparagus Fern", "fine feathery fronds"],
  ["Dumb Cane", "broad variegated leaves"],
  ["Jade Plant", "thick rounded leaves"],
  ["Money Tree", "palmate 5-leaflet"],
  ["Snake Plant", "tall stiff striped blades"],
];
classes.forEach((c, i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = M + col * 3.0, y = 1.85 + row * 1.45;
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 2.75, h: 1.2, fill: { color: PALE }, line: { color: LINEC, width: 1 } });
  s.addText(c[0], { x: x + 0.2, y: y + 0.22, w: 2.4, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 16, bold: true, color: INK });
  s.addText(c[1], { x: x + 0.2, y: y + 0.62, w: 2.4, h: 0.4, margin: 0, fontFace: BODY, fontSize: 12, color: MUTE });
});
s.addText("Chosen for visual distinctness and because I physically own each one (matches the deployment environment).",
  { x: M, y: 5.0, w: 8.7, h: 0.3, margin: 0, fontFace: BODY, fontSize: 11, italic: true, color: MUTE });

// ===================================================================== 4. ARCHITECTURE
s = pres.addSlide();
header(s, "Model architecture", "MobileNetV1 + transfer learning, quantized to INT8");
// pipeline flow
const steps = ["Camera\nframe", "Center-crop\n+ 96×96 RGB", "Rescale\n(baked-in)", "MobileNetV1\nα = 0.25", "Dense → 6\nsoftmax"];
const fw = 1.6, gap = 0.18, fy = 1.75;
let fx = M;
steps.forEach((t, i) => {
  s.addShape(pres.shapes.RECTANGLE, { x: fx, y: fy, w: fw, h: 0.95, fill: { color: i === 3 ? GREEN : PALE }, line: { color: LINEC, width: 1 } });
  s.addText(t, { x: fx, y: fy, w: fw, h: 0.95, margin: 0, align: "center", valign: "middle", fontFace: BODY, fontSize: 11.5, bold: i === 3, color: i === 3 ? WHITE : INK });
  if (i < steps.length - 1) s.addText("→", { x: fx + fw, y: fy, w: gap, h: 0.95, margin: 0, align: "center", valign: "middle", fontFace: BODY, fontSize: 16, color: FAINT });
  fx += fw + gap;
});
bullets(s, [
  "Transfer learning: ImageNet-pretrained MobileNetV1 backbone + a small new head (global-average-pool → dropout → dense-6).",
  "Normalization (0–255 → −1…1) is baked in as the first layer, so the device feeds raw camera pixels.",
  "Post-training INT8 quantization for the deployed model; runs under TensorFlow Lite Micro.",
], M, 3.05, W - 2 * M, { gap: 11 });

// ===================================================================== 5. DECISIONS
s = pres.addSlide();
header(s, "Decisions & tradeoffs", "Choices driven by the 256 KB RAM ceiling");
const dec = [
  ["MobileNetV1, not V2/V3", "V2/V3's channel expansion needs ~293 KB of activations — over budget. V1 has no expansion → ~103 KB. Fits."],
  ["96×96 input", "Higher resolution than the 64/48 I was forced toward on V2; V1's small arena makes 96 affordable."],
  ["Post-training INT8 (not QAT)", "Lossless here (0.82→0.82), ~4× smaller, integer math on the MCU. Simpler than QAT for no accuracy cost."],
  ["QCIF camera, not QQVGA", "The OV7675 driver returns garbage at QQVGA; QCIF is the known-good mode and still fits RAM."],
  ["5-frame peak-hold + threshold", "Rewards the best-aimed frame and rejects low-confidence views → a stable, honest demo."],
  ["TFLM + plain Python, not Edge Impulse", "Full control of the firmware loop and a readable, debuggable pipeline."],
];
dec.forEach((d, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = M + col * 4.45, y = 1.7 + row * 1.18;
  s.addText(d[0], { x, y, w: 4.2, h: 0.32, margin: 0, fontFace: HEAD, fontSize: 14, bold: true, color: GREEN });
  s.addText(d[1], { x, y: y + 0.32, w: 4.2, h: 0.72, margin: 0, fontFace: BODY, fontSize: 11.5, color: INK });
});

// ===================================================================== 6. DATA SOURCES
s = pres.addSlide();
header(s, "Data", "Three sources, general → deployment-specific");
const srcs = [
  ["Online (Kaggle)", "~2,000 imgs", "General plant features — what each species looks like in the wild."],
  ["Phone camera", "my plants", "My actual plants, sharper than the sensor — bridges toward deployment."],
  ["Arduino camera", "600 frames", "Captured through the OV7675 on the demo table — the true deployment domain."],
];
srcs.forEach((c, i) => {
  const x = M + i * 3.0, y = 1.8;
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 2.75, h: 2.7, fill: { color: i === 2 ? GREEN : PALE }, line: { color: LINEC, width: 1 } });
  const fg = i === 2 ? WHITE : INK, fm = i === 2 ? "D7E3D8" : MUTE;
  s.addText(c[0], { x: x + 0.2, y: y + 0.25, w: 2.4, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 16, bold: true, color: fg });
  s.addText(c[1], { x: x + 0.2, y: y + 0.7, w: 2.4, h: 0.4, margin: 0, fontFace: HEAD, fontSize: 22, bold: true, color: fg });
  s.addText(c[2], { x: x + 0.2, y: y + 1.25, w: 2.4, h: 1.3, margin: 0, fontFace: BODY, fontSize: 12.5, color: fm });
});
s.addText("Class imbalance handled with class weights, not by discarding images. Half of the Arduino data is held out for an honest test.",
  { x: M, y: 4.8, w: 8.7, h: 0.5, margin: 0, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE });

// ===================================================================== 7. COLLECTION PIPELINE
s = pres.addSlide();
header(s, "Data collection pipeline", "Capturing the deployment domain through the camera");
bullets(s, [
  "Custom Arduino sketch streams OV7675 frames over USB serial (request/response, sync-header framed).",
  "A Python tool saves frames into per-class folders with a live preview to aim and focus the lens.",
  "~100 frames/plant, moving around each plant on the demo table → realistic, on-device-matched images.",
  "Same camera, lighting, and distance as the demo — the single biggest accuracy lever.",
], M, 1.7, 4.7, { gap: 11 });
placeholder(s, 5.65, 1.7, 3.7, 1.55, "[ capture screenshot 1 ]");
placeholder(s, 5.65, 3.35, 3.7, 1.55, "[ capture screenshot 2 ]");

// ===================================================================== 8. ONLINE + PHONE
s = pres.addSlide();
header(s, "Online & phone data", "Building (and cleaning) the base dataset");
s.addText("Online — Kaggle House Plant Species", { x: M, y: 1.7, w: 4.3, h: 0.35, margin: 0, fontFace: HEAD, fontSize: 15, bold: true, color: GREEN });
bullets(s, [
  "6 of 47 classes selected to match my plants.",
  "Cleaned hidden WebP/MPO files masquerading as .jpg (they crash TF's loader).",
  "EXIF rotation fixed so images aren't fed sideways.",
], M, 2.1, 4.3, { fs: 12.5, gap: 8 });
s.addText("Phone — my own plants", { x: 5.3, y: 1.7, w: 4.0, h: 0.35, margin: 0, fontFace: HEAD, fontSize: 15, bold: true, color: GREEN });
bullets(s, [
  "Photos of the actual plants I own.",
  "Merged into the main dataset as ordinary training data.",
  "Degrade-augmentation (blur, color cast, noise) makes crisp photos look like the grainy sensor feed.",
], 5.3, 2.1, 4.0, { fs: 12.5, gap: 8 });
s.addText("All loading is crop-to-square → resize at load time; source folders are never modified.",
  { x: M, y: 4.95, w: 8.7, h: 0.3, margin: 0, fontFace: BODY, fontSize: 11, italic: true, color: MUTE });

// ===================================================================== 9. TRAINING
s = pres.addSlide();
header(s, "Training strategy", "Three stages: general features → deployment domain");
const stg = [
  ["Stage 1 — Head", "Freeze backbone, train the new 6-class head on the online + phone data."],
  ["Stage 2 — Fine-tune", "Unfreeze the top backbone layers at a low LR to sharpen features."],
  ["Stage 3 — Specialize", "Adapt to the Arduino-camera domain, with safeguards against forgetting."],
];
stg.forEach((d, i) => {
  const x = M + i * 3.0, y = 1.8;
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 2.75, h: 1.5, fill: { color: PALE }, line: { color: LINEC, width: 1 } });
  s.addText(d[0], { x: x + 0.18, y: y + 0.18, w: 2.4, h: 0.35, margin: 0, fontFace: HEAD, fontSize: 14, bold: true, color: INK });
  s.addText(d[1], { x: x + 0.18, y: y + 0.55, w: 2.45, h: 0.9, margin: 0, fontFace: BODY, fontSize: 11.5, color: MUTE });
});
bullets(s, [
  "Forgetting safeguards in Stage 3: freeze early/mid layers, mix in old data (rehearsal), low LR, early-stop.",
  "Class weights for imbalance; degrade-augmentation to close the crisp-vs-camera gap.",
  "Headline metric = accuracy on held-out Arduino frames (predicts the demo).",
], M, 3.55, W - 2 * M, { fs: 13, gap: 9 });

// ===================================================================== 10. RESULTS
s = pres.addSlide();
header(s, "Results", "Float vs INT8 — the four required metrics");
const rows = [
  [{ text: "Variant", options: { bold: true, color: WHITE, fill: { color: GREEN } } },
   { text: "Accuracy (cam)", options: { bold: true, color: WHITE, fill: { color: GREEN } } },
   { text: "Model size", options: { bold: true, color: WHITE, fill: { color: GREEN } } },
   { text: "Latency*", options: { bold: true, color: WHITE, fill: { color: GREEN } } },
   { text: "Tensor arena", options: { bold: true, color: WHITE, fill: { color: GREEN } } }],
  ["Float (reference)", "0.82", "849 KB", "0.27 ms", "—"],
  ["INT8 (deployed)", "0.82", "303 KB", "0.14 ms", "~103 KB"],
];
s.addTable(rows, { x: M, y: 1.8, w: W - 2 * M, h: 1.5, fontFace: BODY, fontSize: 14, color: INK,
  border: { pt: 1, color: LINEC }, align: "left", valign: "middle", rowH: [0.5, 0.5, 0.5], fill: { color: WHITE } });
bullets(s, [
  "INT8 quantization was lossless on the camera test set, at ~2.8× smaller.",
  "~103 KB arena measured on-device — fits the 256 KB RAM alongside the camera buffer.",
  "Held-out Arduino test ≈ 0.82; the real measure is the live demo.",
], M, 3.6, W - 2 * M, { fs: 13, gap: 9 });
s.addText("*Host-side estimate; true on-device latency/RAM read from the Arduino after AllocateTensors().",
  { x: M, y: 5.15, w: 8.7, h: 0.3, margin: 0, fontFace: BODY, fontSize: 10, italic: true, color: FAINT });

// ===================================================================== 11. CODE PRACTICES
s = pres.addSlide();
header(s, "Code practices", "Reproducible, honest, and de-risked");
const cp = [
  ["Single source of truth", "One config.py drives classes, paths, sizes, and hyperparameters."],
  ["Read-only raw data", "The pipeline only ever copies from source folders — never mutates them."],
  ["Idempotent & reproducible", "Splits rebuilt from scratch with a fixed seed; sanitize-on-copy (EXIF/WebP)."],
  ["Separation of concerns", "prepare · train · quantize · firmware · dashboard, each standalone."],
  ["De-risk on hardware early", "An arena-probe sketch measured real RAM before perfecting accuracy."],
  ["Robust serial protocol", "Magic-byte sync framing so the image stream can't desync the dashboard."],
];
cp.forEach((d, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = M + col * 4.45, y = 1.7 + row * 1.15;
  s.addText(d[0], { x, y, w: 4.2, h: 0.3, margin: 0, fontFace: HEAD, fontSize: 14, bold: true, color: GREEN });
  s.addText(d[1], { x, y: y + 0.3, w: 4.2, h: 0.7, margin: 0, fontFace: BODY, fontSize: 11.5, color: INK });
});

// ===================================================================== 12. DEPLOYMENT + DASHBOARD
s = pres.addSlide();
header(s, "Deployment & dashboard", "On-device loop + live serial UI");
bullets(s, [
  "Device loop: capture → center-crop/resize → INT8 inference → 5-frame peak-hold + threshold → serial.",
  "Outputs the most confident recent prediction, or \"No houseplant detected\" below threshold.",
  "Python dashboard shows the model's-eye image and live 6-class confidence bars.",
  "Goes beyond the Serial Monitor — a real local UI for the result.",
], M, 1.7, 4.7, { gap: 11 });
placeholder(s, 5.65, 1.7, 3.7, 3.2, "[ dashboard screenshot / demo video ]");
s.addText("Advanced components: on-hardware camera fine-tuning · INT8 model compression · serial dashboard.",
  { x: M, y: 5.05, w: 8.7, h: 0.3, margin: 0, fontFace: BODY, fontSize: 11, italic: true, color: MUTE });

pres.writeFile({ fileName: "PlantDetector_EE446.pptx" }).then(f => console.log("wrote", f));
