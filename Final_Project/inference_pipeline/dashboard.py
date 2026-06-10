"""Live dashboard for the on-device plant classifier.

Reads the inference sketch's serial stream and shows, in one matplotlib window:
  - the best frame the model saw (IMG packets)
  - a live bar chart of the 6 class confidences (STAT packets)
  - a big prediction label = the 5-frame peak-hold result, or
    "No houseplant detected" when below the on-device confidence threshold.

Serial protocol (must match the Arduino sketch):
  STAT <idx> <conf> <c0> <c1> <c2> <c3> <c4> <c5>\n
       idx  = peak-hold class index, or -1 = below threshold
       conf = peak-hold confidence (0..1)
       c0..c5 = current frame's 6 class scores (0..1)
  IMG <w> <h>\n  followed by  w*h*2 bytes RGB565   (the best frame)

Examples
--------
  python dashboard.py --list
  python dashboard.py --port /dev/cu.usbmodemXXXX
  python dashboard.py --selftest        # validate parsing without hardware
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

LABELS_FILE = Path(__file__).resolve().parent.parent / "models" / "labels.txt"
MAGIC_STAT = "STAT"
MAGIC_IMG = "IMG"
MAGIC_BYTES = bytes([0x55, 0xAA, 0x55, 0xAA])  # precedes the IMG pixel payload (resync)


def load_labels():
    if LABELS_FILE.exists():
        return [l.strip() for l in LABELS_FILE.read_text().splitlines() if l.strip()]
    return [f"class {i}" for i in range(6)]


def rgb565_to_rgb(payload, w, h):
    px = np.frombuffer(payload, dtype=">u2").astype(np.uint32)
    r = ((px >> 11) & 0x1F) << 3
    g = ((px >> 5) & 0x3F) << 2
    b = (px & 0x1F) << 3
    return np.stack([r, g, b], -1).astype(np.uint8).reshape(h, w, 3)


class State:
    """Thread-safe latest-values shared between the reader and the GUI."""
    def __init__(self, n):
        self.lock = threading.Lock()
        self.conf = np.zeros(n)        # current-frame class scores
        self.idx = -1                  # peak-hold class (-1 = uncertain)
        self.peak = 0.0                # peak-hold confidence
        self.image = None              # last best frame (HxWx3)
        self.updated = 0               # bump counter for the GUI

    def set_status(self, idx, peak, conf):
        with self.lock:
            self.idx, self.peak = idx, peak
            self.conf = np.array(conf)
            self.updated += 1

    def set_image(self, img):
        with self.lock:
            self.image = img
            self.updated += 1

    def snapshot(self):
        with self.lock:
            return self.idx, self.peak, self.conf.copy(), self.image, self.updated


class Parser:
    """Feeds raw serial reads; calls state setters when a packet completes.

    Reads control lines as ASCII; on an IMG header it then consumes exactly
    w*h*2 binary bytes via the provided read_exact callback.
    """
    def __init__(self, state, n, read_exact):
        self.state, self.n, self.read_exact = state, n, read_exact

    def handle_line(self, line):
        parts = line.split()
        if not parts:
            return
        if parts[0] == MAGIC_STAT and len(parts) >= 2 + self.n:
            idx = int(parts[1])
            peak = float(parts[2])
            conf = [float(v) for v in parts[3:3 + self.n]]
            self.state.set_status(idx, peak, conf)
        elif parts[0] == MAGIC_IMG and len(parts) >= 3:
            w, h = int(parts[1]), int(parts[2])
            if not (0 < w <= 320 and 0 < h <= 320):
                return
            # Resync on the 0x55AA55AA magic that precedes the pixel payload, so a
            # stray/missing byte between the header line and the data can't shear
            # the image. The magic can't appear in an ASCII STAT line (0xAA).
            window = bytearray()
            for _ in range(16):
                b = self.read_exact(1)
                if not b:
                    return
                window += b
                if len(window) > 4:
                    window = window[-4:]
                if bytes(window) == MAGIC_BYTES:
                    break
            else:
                return  # magic not found -> skip this frame
            payload = self.read_exact(w * h * 2)
            if payload is not None:
                self.state.set_image(rgb565_to_rgb(payload, w, h))


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------
def serial_reader(port, baud, parser, stop):
    import serial
    with serial.Serial(port, baud, timeout=1) as ser:
        time.sleep(2.0)
        ser.reset_input_buffer()

        def read_exact(nbytes):
            buf = bytearray()
            while len(buf) < nbytes and not stop.is_set():
                chunk = ser.read(nbytes - len(buf))
                if not chunk:
                    return None
                buf += chunk
            return bytes(buf)

        parser.read_exact = read_exact
        while not stop.is_set():
            raw = ser.readline()
            if raw:
                parser.handle_line(raw.decode("ascii", "ignore").strip())


# ---------------------------------------------------------------------------
# Synthetic stream for --selftest (no hardware)
# ---------------------------------------------------------------------------
def sim_reader(parser, state, n, stop):
    rng = np.random.default_rng(0)
    t = 0
    while not stop.is_set():
        cls = (t // 20) % n                 # slowly cycle the "true" plant
        conf = rng.random(n) * 0.2
        conf[cls] = 0.5 + rng.random() * 0.5
        conf = conf / conf.sum()
        peak = float(conf[cls])
        idx = cls if peak >= 0.6 else -1
        parser.handle_line(f"STAT {idx} {peak:.3f} " + " ".join(f"{c:.3f}" for c in conf))
        if t % 5 == 0:                      # occasional fake image
            img = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
            state.set_image(img)
        t += 1
        time.sleep(0.1)


# ---------------------------------------------------------------------------
def run_gui(state, labels, stop):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n = len(labels)
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 5))
    fig.canvas.manager.set_window_title("Plant Detector")
    blank = np.zeros((96, 96, 3), np.uint8)
    im = ax_img.imshow(blank)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title("best frame")
    short = [l.split(" (")[0] for l in labels]
    bars = ax_bar.barh(range(n), [0] * n)
    ax_bar.set_yticks(range(n)); ax_bar.set_yticklabels(short)
    ax_bar.set_xlim(0, 1); ax_bar.invert_yaxis()
    ax_bar.set_xlabel("confidence")
    title = fig.suptitle("waiting...", fontsize=16)

    def update(_):
        idx, peak, conf, img, _u = state.snapshot()
        if img is not None:
            im.set_data(img)
        cur_best = int(np.argmax(conf)) if len(conf) else -1
        for i, b in enumerate(bars):
            b.set_width(conf[i])
            # green only if this class is BOTH the current top bar AND the
            # peak-hold detection -- so a held label stops being green the
            # moment its bar is no longer the highest.
            b.set_color("tab:green" if (i == idx and i == cur_best) else "tab:blue")
        if idx < 0:
            title.set_text("No houseplant detected")
        else:
            title.set_text(f"{short[idx]}   {peak*100:.0f}%")
        title.set_color("black")
        return [im, title, *bars]

    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    try:
        plt.show()
    finally:
        stop.set()


def selftest(labels):
    """Validate the parser end-to-end on synthetic packets (no GUI)."""
    n = len(labels)
    state = State(n)
    parser = Parser(state, n, read_exact=lambda k: None)
    # STAT
    parser.handle_line("STAT 2 0.873 0.02 0.05 0.87 0.02 0.02 0.02")
    idx, peak, conf, _, _ = state.snapshot()
    assert idx == 2 and abs(peak - 0.873) < 1e-6 and abs(conf.sum() - 1.0) < 0.01, "STAT parse"
    # uncertain
    parser.handle_line("STAT -1 0.41 0.2 0.2 0.2 0.1 0.2 0.1")
    assert state.snapshot()[0] == -1, "threshold/uncertain"
    # IMG: feed a stateful stream of [magic + payload], like the real device.
    w, h = 4, 3
    payload = bytes(np.random.default_rng(0).integers(0, 256, w * h * 2, dtype=np.uint8))
    stream = MAGIC_BYTES + payload
    pos = [0]
    def recv(k):
        chunk = stream[pos[0]:pos[0] + k]
        pos[0] += k
        return chunk if chunk else None
    parser.read_exact = recv
    parser.handle_line(f"IMG {w} {h}")
    assert state.snapshot()[3].shape == (h, w, 3), "IMG decode"
    print(f"selftest OK  (labels={n}: {', '.join(s.split(' (')[0] for s in labels)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sim", action="store_true", help="run GUI with fake data")
    args = ap.parse_args()

    labels = load_labels()
    n = len(labels)

    if args.list:
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"  {p.device:<28} {p.description}")
        return
    if args.selftest:
        selftest(labels)
        return

    state = State(n)
    stop = threading.Event()
    parser = Parser(state, n, read_exact=lambda k: None)

    if args.sim:
        t = threading.Thread(target=sim_reader, args=(parser, state, n, stop), daemon=True)
    else:
        if not args.port:
            ap.error("--port required (or use --list / --sim / --selftest)")
        t = threading.Thread(target=serial_reader,
                             args=(args.port, args.baud, parser, stop), daemon=True)
    t.start()
    run_gui(state, labels, stop)


if __name__ == "__main__":
    main()
