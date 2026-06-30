#!/usr/bin/env python3
"""
AI Book Reader - RapidOCR (ONNX) + Piper TTS
=============================================
1. Captures a high-quality image from the webcam.
2. Enhances the image (denoise -> upscale -> CLAHE -> sharpen).
3. Extracts text using RapidOCR (PaddleOCR models running safely on ONNX Runtime).
   *This completely fixes the PaddlePaddle C++ Segmentation Fault on Pi 5.*
4. Speaks the extracted text using Piper TTS (Amy - natural female voice).

Install on Raspberry Pi:
    pip3 install rapidocr-onnxruntime piper-tts sounddevice soundfile numpy opencv-python --break-system-packages
    sudo apt install -y libportaudio2

Run:
    python3 ocr_reader.py --device 0
"""

import argparse
import io
import os
import sys
import time
import unicodedata
import wave
import cv2
import numpy as np

# ---------------------------------------------
# RapidOCR Import (Safe alternative to PaddleOCR)
# ---------------------------------------------
try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("ERROR: rapidocr_onnxruntime is not installed.")
    print("  Fix: pip3 install rapidocr-onnxruntime --break-system-packages")
    sys.exit(1)

# ---------------------------------------------
# Piper TTS Import
# ---------------------------------------------
try:
    from piper.voice import PiperVoice
    import sounddevice as sd
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("Warning: piper-tts or sounddevice not installed - will fall back to espeak.")
    print("  Fix: pip3 install piper-tts sounddevice soundfile --break-system-packages")


# ===============================================
# SECTION 1 - CAMERA
# ===============================================

def capture_image(device: int, output_path: str,
                  width: int = 1920, height: int = 1080) -> bool:
    """Capture one high-quality frame with hardware auto-focus enabled."""
    print(f"Opening camera /dev/video{device}...")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open camera device {device}.")
        return False

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)
    
    print("Flushing camera buffer and auto-focusing (60 frames)...")
    # Read and discard 60 frames to ensure we get a fresh, focused image 
    # and NOT an old buffered frame from a previous run.
    frame = None
    for i in range(60):
        ok, frame = cap.read()
        if not ok:
            print("  Camera stream interrupted.")
            break
        # Small sleep helps the hardware auto-focus motor physically move
        time.sleep(0.03)

    cap.release()

    if not ok or frame is None:
        print("  ERROR: Failed to read fresh frame from camera.")
        return False

    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"  Captured -> {output_path}")
    return True


# ===============================================
# SECTION 2 - IMAGE ENHANCEMENT PIPELINE
# ===============================================

def _deskew(img: np.ndarray) -> np.ndarray:
    """Detect and correct small rotation angles in the image."""
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img  
    angle  = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:
        return img  
    h, w   = img.shape[:2]
    M      = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_for_ocr(image_path: str, output_path: str) -> str:
    """Enhance image for OCR."""
    img = cv2.imread(image_path)
    if img is None:
        print("  WARNING: Could not read image for enhancement. Using original.")
        return image_path

    h, w = img.shape[:2]

    # 1. Upscale
    min_width = 2400
    if w < min_width:
        scale = min_width / w
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LANCZOS4)
        print(f"  Upscaled from {w}x{h} -> {img.shape[1]}x{img.shape[0]}")

    # 2. Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                          templateWindowSize=7, searchWindowSize=21)

    # 3. CLAHE
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Sharpen
    blur    = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    img     = cv2.addWeighted(img, 1.65, blur, -0.65, 0)

    # 5. Deskew
    img = _deskew(img)

    # 6. Padding
    img = cv2.copyMakeBorder(img, 60, 60, 60, 60,
                             cv2.BORDER_CONSTANT, value=[255, 255, 255])

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"  Enhanced image saved -> {output_path}")
    return output_path


# ===============================================
# SECTION 3 - RAPIDOCR TEXT EXTRACTION
# ===============================================

_ocr_instance: "RapidOCR | None" = None

def _get_ocr() -> "RapidOCR":
    global _ocr_instance
    if _ocr_instance is None:
        print("Loading ONNX OCR model (safe, no segfaults)...")
        _ocr_instance = RapidOCR()
    return _ocr_instance


def extract_text(image_path: str, min_confidence: float = 0.50) -> str:
    """Run RapidOCR on the enhanced image."""
    if not OCR_AVAILABLE:
        print("OCR not available.")
        return ""

    ocr = _get_ocr()

    def _run(path: str) -> list[tuple[str, float]]:
        try:
            result, _ = ocr(path)
            items = []
            if result:
                for box, text, score in result:
                    if text.strip():
                        items.append((text.strip(), float(score)))
            return items
        except Exception as e:
            print(f"  OCR run error: {e}")
            return []

    print("Running OCR on enhanced image...")
    items = _run(image_path)

    if not items:
        print("  No text found - retrying with 180 degree rotation...")
        img     = cv2.imread(image_path)
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        rot_path = image_path.replace(".jpg", "_rot180.jpg")
        cv2.imwrite(rot_path, rotated)
        items = _run(rot_path)
        try:
            os.remove(rot_path)
        except OSError:
            pass

    if not items:
        print("  No text detected in image.")
        return ""

    print(f"\n{'-'*42}")
    print(f"  {'TEXT':<30}  CONF")
    print(f"{'-'*42}")
    accepted: list[str] = []
    for text, conf in items:
        marker = "[OK]" if conf >= min_confidence else "[XX]"
        print(f"  {marker} {text:<30}  {conf:.0%}")
        if conf >= min_confidence:
            accepted.append(text)
    print(f"{'-'*42}")

    full_text = " ".join(accepted)
    print(f"\n--- Extracted Text ---\n{full_text}\n{'-'*22}")
    return full_text


# ===============================================
# SECTION 4 - TEXT SANITISER
# ===============================================

def sanitize_for_speech(text: str) -> str:
    if not text:
        return "No text detected."

    cleaned = unicodedata.normalize("NFKD", text)
    substitutions = {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2014": ", ", "\u2013": "-", "\u2026": "...", "\u2022": ",",
        "\u25c6": "", "\u00a0": " ", "\u00b7": ".", "|": ""
    }
    for src, dst in substitutions.items():
        cleaned = cleaned.replace(src, dst)

    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip() or "No text detected."


# ===============================================
# SECTION 5 - PIPER TTS
# ===============================================

PIPER_MODEL_DIR = os.path.expanduser("~/pi-cam-feed/piper_models")
PIPER_ONNX      = os.path.join(PIPER_MODEL_DIR, "en_US-amy-medium.onnx")
PIPER_ONNX_JSON = os.path.join(PIPER_MODEL_DIR, "en_US-amy-medium.onnx.json")
_HF_BASE        = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"


def ensure_piper_models() -> bool:
    if os.path.exists(PIPER_ONNX) and os.path.exists(PIPER_ONNX_JSON):
        return True
    print("Piper model not found - downloading Amy voice (~30 MB, one-time)...")
    os.makedirs(PIPER_MODEL_DIR, exist_ok=True)
    ok = True
    for fname, dest in [
        ("en_US-amy-medium.onnx", PIPER_ONNX),
        ("en_US-amy-medium.onnx.json", PIPER_ONNX_JSON),
    ]:
        ret = os.system(f'wget -q --show-progress -O "{dest}" "{_HF_BASE}/{fname}"')
        if ret != 0:
            print(f"  Download failed: {fname}")
            ok = False
    return ok


def speak_text(text: str) -> None:
    clean = sanitize_for_speech(text)
    preview = clean[:80] + ("..." if len(clean) > 80 else "")
    print(f'\nSpeaking: "{preview}"')

    safe = clean.replace('"', "'")
    temp_wav = os.path.expanduser("~/pi-cam-feed/temp_speech.wav")
    
    # We use the piper command-line tool exactly as you proved it works!
    # This completely bypasses the Python API bugs.
    if os.path.exists(PIPER_ONNX):
        cmd = f'echo "{safe}" | piper --model "{PIPER_ONNX}" --output_file "{temp_wav}" 2>/dev/null'
        os.system(cmd)
        
        # Play the generated WAV file
        os.system(f'paplay "{temp_wav}" 2>/dev/null || aplay -q "{temp_wav}"')
        return
    else:
        print("Piper model not found! Using espeak fallback...")
        os.system(f'espeak -s 120 -a 180 -v en-us "{safe}"')


# ===============================================
# SECTION 6 - MAIN
# ===============================================

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Book Reader - RapidOCR + Piper TTS")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("--confidence", type=float, default=0.50, help="Min OCR confidence")
    parser.add_argument("--width", type=int, default=1920, help="Capture width")
    parser.add_argument("--height", type=int, default=1080, help="Capture height")
    args = parser.parse_args()

    raw_path      = os.path.expanduser("~/pi-cam-feed/temp_capture.jpg")
    enhanced_path = os.path.expanduser("~/pi-cam-feed/temp_enhanced.jpg")

    # -- Step 1: Capture ----------------------------------------------
    print("\n[ 1/4 ] Capturing image from camera...")
    if not capture_image(args.device, raw_path, args.width, args.height):
        speak_text("Camera error. Please check the connection.")
        sys.exit(1)

    # -- Step 2: Enhance ----------------------------------------------
    print("\n[ 2/4 ] Enhancing image for OCR...")
    enhanced_path = enhance_for_ocr(raw_path, enhanced_path)

    # -- Step 3: OCR --------------------------------------------------
    print("\n[ 3/4 ] Extracting text with ONNX OCR...")
    text = extract_text(enhanced_path, min_confidence=args.confidence)

    # -- Step 4: Speak ------------------------------------------------
    print("\n[ 4/4 ] Speaking with Piper TTS (Amy)...")
    speak_text(text if text.strip() else "No text was detected in the image.")


if __name__ == "__main__":
    main()
