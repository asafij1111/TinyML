// Standalone slide: detailed training/test accuracy tables (matches deck style).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Accuracy detail";

const INK = "1F2937", MUTE = "6B7280", FAINT = "9CA3AF";
const GREEN = "2C5F2D", PALE = "F4F6F4", LINEC = "E5E7EB", WHITE = "FFFFFF";
const HEAD = "Helvetica Neue", BODY = "Helvetica Neue";
const W = 10, M = 0.65;

const s = pres.addSlide();
s.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.5, w: 0.14, h: 0.14, fill: { color: GREEN }, line: { type: "none" } });
s.addText("RESULTS", { x: M + 0.26, y: 0.43, w: 8, h: 0.3, margin: 0, fontFace: BODY, fontSize: 11, bold: true, color: GREEN, charSpacing: 3 });
s.addText("Training & test accuracy", { x: M, y: 0.75, w: W - 2 * M, h: 0.6, margin: 0, fontFace: HEAD, fontSize: 28, bold: true, color: INK });

function hcell(t) { return { text: t, options: { bold: true, color: WHITE, fill: { color: GREEN }, align: "left", valign: "middle" } }; }
function cell(t, b) { return { text: t, options: { color: INK, bold: !!b, align: "left", valign: "middle" } }; }

// ---- Table 1: accuracy by training phase ----
s.addText("Accuracy through the pipeline", { x: M, y: 1.45, w: 8, h: 0.3, margin: 0, fontFace: HEAD, fontSize: 14, bold: true, color: GREEN });
s.addTable([
  [hcell("Training phase"), hcell("Val"), hcell("General test"), hcell("Camera test")],
  [cell("Stage 1 — General training (online + phone)"), cell("77%"), cell("76.9%"), cell("23.3%")],
  [cell("Stage 2 — Camera fine-tuning (Arduino)"), cell("85%"), cell("69.4%"), cell("81.7%", true)],
], { x: M, y: 1.78, w: W - 2 * M, colW: [4.7, 1.3, 1.65, 1.05], rowH: [0.42, 0.5, 0.5],
  fontFace: BODY, fontSize: 13, border: { pt: 1, color: LINEC }, fill: { color: WHITE }, valign: "middle" });

// ---- Table 2: deployment (quantization) ----
s.addText("Deployed model (quantization)", { x: M, y: 3.45, w: 8, h: 0.3, margin: 0, fontFace: HEAD, fontSize: 14, bold: true, color: GREEN });
s.addTable([
  [hcell("Model"), hcell("Camera test"), hcell("General test"), hcell("Size"), hcell("Latency*")],
  [cell("Float (reference)"), cell("81.7%"), cell("69.4%"), cell("849 KB"), cell("0.27 ms")],
  [cell("INT8 (on-device)", true), cell("81.7%", true), cell("71.5%"), cell("303 KB", true), cell("0.14 ms")],
], { x: M, y: 3.78, w: W - 2 * M, colW: [2.6, 1.85, 1.75, 1.1, 1.4], rowH: [0.42, 0.5, 0.5],
  fontFace: BODY, fontSize: 13, border: { pt: 1, color: LINEC }, fill: { color: WHITE }, valign: "middle" });

s.addText("Camera test = held-out OV7675 frames from the same capture sessions (correlated → optimistic; the live demo is the true measure).   *host-side latency estimate.",
  { x: M, y: 5.15, w: W - 2 * M, h: 0.4, margin: 0, fontFace: BODY, fontSize: 9.5, italic: true, color: FAINT });

pres.writeFile({ fileName: "accuracy_table_slide.pptx" }).then(f => console.log("wrote", f));
