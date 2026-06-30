#!/usr/bin/env python3
"""
Live Camera Feed Server for Raspberry Pi 5 + Arducam UVC 573
=============================================================
Streams MJPEG video from a UVC camera over HTTP.
Access the feed from any browser on the same network.

Usage:
    python3 stream_server.py [--port 8080] [--device 0] [--width 1920] [--height 1080]
"""

import argparse
import time
import threading
import cv2
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------- Camera wrapper (thread-safe) ----------

class CameraStream:
    """Thread-safe camera capture using OpenCV."""

    def __init__(self, device: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self._thread = None
        self.frame_count = 0
        self.start_time = time.time()
        self.actual_fps = 0.0

    def start(self):
        """Open the camera and start the capture thread."""
        import platform
        is_windows = platform.system() == "Windows"

        if is_windows:
            # Try DirectShow first on Windows (avoids MSMF frame grab failures)
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.device)
        else:
            # Linux / Pi fallback
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.device)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self.device}. "
                "Check that the camera is connected and detected."
            )

        # Force MJPG codec to support high resolution/FPS over USB
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read actual negotiated values
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[camera] Opened /dev/video{self.device}  "
              f"{actual_w}x{actual_h} @ {actual_fps:.1f} fps")

        self.running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                # Reset counters every 5 seconds to keep fps measurement fresh
                if elapsed > 5:
                    self.frame_count = 0
                    self.start_time = time.time()

    def get_frame(self):
        """Return the latest JPEG-encoded frame bytes, or None."""
        with self.lock:
            if self.frame is None:
                return None
            # Encode as JPEG with decent quality
            ok, buf = cv2.imencode(".jpg", self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            return buf.tobytes()

    def get_stats(self):
        with self.lock:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0
            return {
                "resolution": f"{w}x{h}",
                "fps": round(self.actual_fps, 1),
                "device": f"/dev/video{self.device}",
                "status": "streaming" if self.running else "stopped",
            }

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()


# Global camera instance (initialized in main)
camera: CameraStream = None  # type: ignore


# ---------- Routes ----------

@app.route("/")
def index():
    return render_template("index.html")


def generate_mjpeg():
    """Yield MJPEG frames as a multipart HTTP response."""
    while True:
        frame_bytes = camera.get_frame()
        if frame_bytes is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        # Throttle to ~30 fps max on the HTTP side
        time.sleep(0.033)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    return jsonify(camera.get_stats())


@app.route("/api/snapshot")
def api_snapshot():
    frame_bytes = camera.get_frame()
    if frame_bytes is None:
        return "No frame available", 503
    return Response(frame_bytes, mimetype="image/jpeg")


# ---------- Main ----------

def main():
    global camera

    parser = argparse.ArgumentParser(description="Pi Camera Live Feed Server")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    parser.add_argument("--device", type=int, default=0, help="Video device index (default 0)")
    parser.add_argument("--width", type=int, default=1920, help="Capture width (default 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Capture height (default 1080)")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS (default 30)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    args = parser.parse_args()

    camera = CameraStream(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    print("=" * 60)
    print("  Pi Camera Live Feed Server")
    print("=" * 60)
    print(f"  Camera device : /dev/video{args.device}")
    print(f"  Resolution    : {args.width}x{args.height}")
    print(f"  Target FPS    : {args.fps}")
    print(f"  Server        : http://{args.host}:{args.port}")
    print("=" * 60)

    try:
        camera.start()
        app.run(host=args.host, port=args.port, threaded=True)
    except KeyboardInterrupt:
        print("\n[server] Shutting down…")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
