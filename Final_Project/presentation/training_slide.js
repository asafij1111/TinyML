// Standalone single slide: training framed as TWO stages (matches the deck style).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Training strategy (2 stages)";

const INK = "1F2937", MUTE = "6B7280", FAINT = "9CA3AF";
const GREEN = "2C5F2D", PALE = "F4F6F4", LINEC = "E5E7EB", WHITE = "FFFFFF";
const HEAD = "Helvetica Neue", BODY = "Helvetica Neue";
const W = 10, M = 0.65;

const s = pres.addSlide();

// header (kicker + title) — same as the rest of the deck
s.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.55, w: 0.14, h: 0.14, fill: { color: GREEN }, line: { type: "none" } });
s.addText("TRAINING STRATEGY", { x: M + 0.26, y: 0.48, w: 8, h: 0.3, margin: 0,
  fontFace: BODY, fontSize: 11, bold: true, color: GREEN, charSpacing: 3 });
s.addText("Two stages: general training → on-device fine-tuning", { x: M, y: 0.82, w: W - 2 * M, h: 0.7,
  margin: 0, fontFace: HEAD, fontSize: 30, bold: true, color: INK });

// ---- card helper ----
function card(x, label, title, sub, items, dark) {
  const y = 1.85, w = 4.05, h = 2.95;
  const bg = dark ? GREEN : PALE, fg = dark ? WHITE : INK, fm = dark ? "D7E3D8" : MUTE;
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: bg }, line: { color: LINEC, width: 1 } });
  s.addText(label, { x: x + 0.25, y: y + 0.22, w: w - 0.5, h: 0.28, margin: 0,
    fontFace: BODY, fontSize: 11, bold: true, color: dark ? "BFE0C2" : GREEN, charSpacing: 3 });
  s.addText(title, { x: x + 0.25, y: y + 0.5, w: w - 0.5, h: 0.4, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: fg });
  s.addText(sub, { x: x + 0.25, y: y + 0.92, w: w - 0.5, h: 0.3, margin: 0,
    fontFace: BODY, fontSize: 12.5, italic: true, color: fm });
  s.addText(items.map((t, i) => ({ text: t, options: {
    bullet: { indent: 13 }, breakLine: i < items.length - 1,
    fontFace: BODY, fontSize: 12, color: fg, paraSpaceAfter: 7 } })),
    { x: x + 0.25, y: y + 1.32, w: w - 0.5, h: 1.5, margin: 0, valign: "top" });
}

card(M, "STAGE 1", "General training", "Online + phone images", [
  "Pretrained MobileNetV1 backbone (ImageNet transfer learning).",
  "Train the new 6-class head, then unfreeze the top layers and fine-tune at low LR.",
  "Learns what the six species look like in general.",
], false);

// arrow between the two cards
s.addText("→", { x: 4.72, y: 1.85, w: 0.56, h: 2.95, margin: 0, align: "center", valign: "middle",
  fontFace: BODY, fontSize: 26, color: FAINT });

card(5.3, "STAGE 2", "Environment-specific fine-tuning", "Arduino-camera data", [
  "Fine-tune on frames captured through the OV7675 on the demo table.",
  "Adapts to real deployment: low-res sensor, lighting, distance.",
  "Rehearsal + early-stopping → specialize without forgetting.",
], true);

s.addText("Stage 2 — on-device camera fine-tuning — is what makes the model work on the actual hardware.",
  { x: M, y: 5.0, w: W - 2 * M, h: 0.35, margin: 0, fontFace: BODY, fontSize: 11.5, italic: true, color: MUTE });

pres.writeFile({ fileName: "training_2stage_slide.pptx" }).then(f => console.log("wrote", f));
