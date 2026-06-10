"""capture_serial.py - save OV7675 frames streamed by plant_capture.ino.

Pairs with the plant_capture.ino sketch. Requests frames one at a time over USB
serial, decodes RGB565 -> RGB, and saves PNGs into one folder per plant. Run it
once per plant while you slowly move the camera around it.

Protocol (must match the sketch):
  Mac -> board : 'c'                         (request one frame)
  board -> Mac : 0x55 0xAA 0x55 0xAA         (sync MAGIC)
                 width  (big-endian uint16)
                 height (big-endian uint16)
                 width*height*2 bytes RGB565 (big-endian per pixel)

Examples
--------
  # list available serial ports
  python capture_serial.py --list

  # capture 100 frames of the aloe vera, ~3 fps, into collected_arduino/aloe_vera/
  python capture_serial.py --port /dev/cu.usbmodemXXXX --plant aloe_vera --num 100

Tip: use the same class folder names as config.COLLECTED_ALIASES keys
(aloe_vera, asparagus_fern, dumb_cane, jade_plant, money_tree, snake_plant) so
the captures drop straight into the training pipeline later.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial not installed. Run:  pip install pyserial")

MAGIC = bytes([0x55, 0xAA, 0x55, 0xAA])
DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / "collected_arduino"


class LivePreview:
    """A single matplotlib window that refreshes in place (no new windows).

    Used both for the lens-focus preview and for live feedback during capture.
    Created lazily on first frame so we know the image size.
    """

    def __init__(self, title):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()                              # interactive: don't block
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title("plant_capture")
        self.ax.set_title(title)
        self.ax.axis("off")
        self.im = None
        self.fig.show()

    def update(self, img, subtitle=None):
        if self.im is None:
            self.im = self.ax.imshow(img)
        else:
            self.im.set_data(img)
        if subtitle is not None:
            self.ax.set_title(subtitle)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()         # repaint the SAME window

    def alive(self):
        return self.plt.fignum_exists(self.fig.number)

    def close(self):
        self.plt.close(self.fig)


def list_serial_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found. Is the board plugged in?")
    for p in ports:
        print(f"  {p.device:<28} {p.description}")


def read_exact(ser, n):
    """Read exactly n bytes or raise on timeout."""
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError(f"serial timeout (got {len(buf)}/{n} bytes)")
        buf += chunk
    return bytes(buf)


def sync_to_magic(ser):
    """Read byte-by-byte until the 4-byte MAGIC is seen (resync each frame)."""
    window = bytearray()
    while True:
        b = ser.read(1)
        if not b:
            raise TimeoutError("serial timeout while waiting for frame header")
        window += b
        if len(window) > 4:
            window = window[-4:]
        if window == MAGIC:
            return


def rgb565_to_rgb(payload, w, h):
    """Decode big-endian RGB565 bytes -> (h, w, 3) uint8 RGB array."""
    px = np.frombuffer(payload, dtype=">u2").astype(np.uint32)
    r = ((px >> 11) & 0x1F) << 3
    g = ((px >> 5) & 0x3F) << 2
    b = (px & 0x1F) << 3
    return np.stack([r, g, b], axis=-1).astype(np.uint8).reshape(h, w, 3)


def grab_frame(ser):
    ser.write(b"c")
    sync_to_magic(ser)
    w = int.from_bytes(read_exact(ser, 2), "big")
    h = int.from_bytes(read_exact(ser, 2), "big")
    if not (0 < w <= 640 and 0 < h <= 480):
        raise ValueError(f"implausible frame size {w}x{h} -- serial desync?")
    payload = read_exact(ser, w * h * 2)
    return rgb565_to_rgb(payload, w, h)


def next_index(out_dir):
    """Continue numbering if the folder already has captures."""
    existing = [int(p.stem) for p in out_dir.glob("*.png") if p.stem.isdigit()]
    return max(existing) + 1 if existing else 0


def run_preview(ser):
    """Live focus preview: stream frames into one in-place window, save nothing.

    Rotate the camera lens until the image looks sharp, then close the window
    (or Ctrl-C) and run a real capture.
    """
    pv = LivePreview("FOCUS preview - adjust lens, then close window to stop")
    print("Live preview running. Adjust the lens for sharpness; "
          "close the window (or Ctrl-C) when done.")
    n = 0
    try:
        while pv.alive():
            try:
                img = grab_frame(ser)
            except (TimeoutError, ValueError):
                continue
            n += 1
            pv.update(img, subtitle=f"FOCUS preview - frame {n} ({img.shape[1]}x{img.shape[0]})")
    except KeyboardInterrupt:
        pass
    finally:
        pv.close()
    print(f"Preview ended ({n} frames shown).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--preview", action="store_true",
                    help="live focus preview only (no saving); adjust the lens, then close window")
    ap.add_argument("--port", help="serial port, e.g. /dev/cu.usbmodem1101")
    ap.add_argument("--plant", help="class folder name, e.g. aloe_vera")
    ap.add_argument("--num", type=int, default=100, help="frames to capture (default 100)")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds between frames; pace your camera movement (default 0.3)")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="disable the live window during capture (faster)")
    ap.add_argument("--preview-secs", type=float, default=4.0,
                    help="seconds of live preview before saving starts, to confirm "
                         "focus/framing (default 4; 0 to skip)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help=f"base output dir (default {DEFAULT_OUTDIR})")
    ap.add_argument("--warmup", type=int, default=5,
                    help="throwaway frames so auto-exposure settles (default 5)")
    args = ap.parse_args()

    if args.list:
        list_serial_ports()
        return
    if not args.port:
        ap.error("--port is required (use --list to find the port)")
    if not args.preview and not args.plant:
        ap.error("--plant is required when capturing (omit it only with --preview)")

    print(f"Opening {args.port} @ {args.baud}...")
    with serial.Serial(args.port, args.baud, timeout=3) as ser:
        time.sleep(2.0)              # let the board reset after port opens
        ser.reset_input_buffer()

        for _ in range(args.warmup):  # discard auto-exposure settling frames
            try:
                grab_frame(ser)
            except (TimeoutError, ValueError):
                pass

        # ---- Focus preview mode: live window, no saving ----
        if args.preview:
            run_preview(ser)
            return

        # ---- Capture mode ----
        out_dir = Path(args.outdir) / args.plant
        out_dir.mkdir(parents=True, exist_ok=True)
        pv = LivePreview(f"capturing {args.plant}") if args.show else None

        # ---- Pre-capture preview: confirm focus/framing before saving ----
        if pv is not None and args.preview_secs > 0:
            print(f"Preview for {args.preview_secs:.0f}s -- confirm focus/framing; "
                  f"saving starts after (close window to abort).")
            t_end = time.time() + args.preview_secs
            while time.time() < t_end and pv.alive():
                try:
                    img = grab_frame(ser)
                except (TimeoutError, ValueError):
                    continue
                pv.update(img, subtitle=f"PREVIEW {args.plant} -- saving in {t_end - time.time():.0f}s")
            if not pv.alive():
                print("  preview window closed -- aborting before capture.")
                return

        start = next_index(out_dir)
        print(f"Capturing {args.num} frames -> {out_dir}  (start index {start})")
        print("Move the camera slowly around the plant...")
        saved = 0
        for i in range(args.num):
            if pv is not None and not pv.alive():     # user closed the window -> stop early
                print("  preview window closed -- stopping.")
                break
            try:
                img = grab_frame(ser)
            except (TimeoutError, ValueError) as e:
                print(f"  frame {i}: {e} (skipped)")
                continue
            idx = start + saved
            Image.fromarray(img).save(out_dir / f"{idx:04d}.png")
            saved += 1
            if pv is not None:
                pv.update(img, subtitle=f"{args.plant}  saved {saved}/{args.num}")
            if saved % 10 == 0:
                print(f"  saved {saved}/{args.num}")
            time.sleep(args.delay)

        if pv is not None:
            pv.close()
        print(f"Done. {saved} images in {out_dir}")


if __name__ == "__main__":
    main()
