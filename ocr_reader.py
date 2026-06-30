#!/usr/bin/env python3
"""
AI Book Reader - Hybrid RapidOCR + Moondream2 Pipeline
=======================================================
Architecture:
  1. Camera captures image with auto-focus
  2. OpenCV enhances image (CLAHE + Sharpen + Upscale + Deskew)
  3. RapidOCR (ONNX) extracts text at ~95% accuracy (no segfault)
  4. Piper TTS reads the extracted text aloud
  5. Moondream2 explains/summarizes using extracted text as context
  6. All interactions are voice-guided with Y/N keyboard input

Features:
  - Voice-guided interactions (all questions spoken aloud)
  - Y/N single key press (no Enter needed)
  - SPACEBAR = Pause / Resume at any time
  - Document alignment check before capture
  - Page memory with FIFO flush warning
  - Auto-ready camera between pages

Install:
    pip3 install rapidocr-onnxruntime requests opencv-python numpy --break-system-packages
    ollama run moondream

Run:
    python3 book_reader.py --device 0
"""

import argparse
import base64
import json
import os
import sys
import time
import tty
import termios
import unicodedata
import requests
import cv2
import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────
MAX_MEMORY_PAGES = 10
WARN_MEMORY_AT   = 8
PIPER_MODEL_DIR  = os.path.expanduser("~/pi-cam-feed/piper_models")
PIPER_ONNX       = os.path.join(PIPER_MODEL_DIR, "en_US-amy-medium.onnx")
TEMP_DIR         = os.path.expanduser("~/pi-cam-feed")
SESSION_FILE     = os.path.join(TEMP_DIR, "session.json")

# ─── RapidOCR (loaded once) ────────────────────────────────
try:
    from rapidocr_onnxruntime import RapidOCR
    _ocr = RapidOCR()
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("WARNING: rapidocr_onnxruntime not installed.")
    print("  Fix: pip3 install rapidocr-onnxruntime --break-system-packages")

# ─── STATE ────────────────────────────────────────────────
page_memory: list[dict] = []
current_page: int        = 0


# ═══════════════════════════════════════════════
# SESSION PERSISTENCE
# ═══════════════════════════════════════════════

def save_session() -> None:
    """Save current page memory and page number to disk after every page."""
    data = {"current_page": current_page, "pages": page_memory}
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Session saved -> {SESSION_FILE} ({len(page_memory)} pages)")


def load_session() -> bool:
    """Load session from disk. Returns True if a valid session was found."""
    global page_memory, current_page
    if not os.path.exists(SESSION_FILE):
        return False
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
        page_memory  = data.get("pages", [])
        current_page = data.get("current_page", 0)
        return len(page_memory) > 0
    except Exception as e:
        print(f"  Could not load session: {e}")
        return False


def delete_session() -> None:
    """Delete the saved session file from disk."""
    global page_memory, current_page
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
        print(f"  Session file deleted: {SESSION_FILE}")
    page_memory  = []
    current_page = 0


# ═══════════════════════════════════════════════
# SECTION 1 - KEYBOARD INPUT
# ═══════════════════════════════════════════════

def get_single_key() -> str:
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def wait_for_yn(question_text: str) -> bool:
    """Speak the question then wait for Y or N. Spacebar triggers pause."""
    speak_text(question_text)
    print(f"\n  {question_text}")
    print("  Press  Y  or  N  (SPACEBAR to pause)")
    while True:
        key = get_single_key().lower()
        if key == 'y':
            print("  -> Yes")
            return True
        elif key == 'n':
            print("  -> No")
            return False
        elif key == ' ':
            handle_pause()
        else:
            print("  (Press Y or N)")


def handle_pause() -> None:
    speak_text("Paused. Press spacebar when you are ready to continue.")
    print("\n========== PAUSED ==========")
    while True:
        if get_single_key() == ' ':
            speak_text("Resuming.")
            print("========== RESUMED ==========\n")
            return


# ═══════════════════════════════════════════════
# SECTION 2 - PIPER TTS
# ═══════════════════════════════════════════════

def sanitize_for_speech(text: str) -> str:
    if not text:
        return "No text detected."
    cleaned = unicodedata.normalize("NFKD", text)
    subs = {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2014": ",", "\u2013": "-", "\u2026": "...", "\u2022": ",",
        "\u00a0": " ", "\u00b7": ".", "|": "", "\n": " ", "\r": ""
    }
    for src, dst in subs.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return " ".join(cleaned.split()).strip() or "No text detected."


def speak_text(text: str) -> None:
    clean = sanitize_for_speech(text)
    if not clean or clean == "No text detected.":
        return
    safe     = clean.replace('"', "'")
    temp_wav = os.path.join(TEMP_DIR, "temp_speech.wav")
    if os.path.exists(PIPER_ONNX):
        os.system(f'echo "{safe}" | piper --model "{PIPER_ONNX}" '
                  f'--output_file "{temp_wav}" 2>/dev/null')
        os.system(f'paplay "{temp_wav}" 2>/dev/null || aplay -q "{temp_wav}" 2>/dev/null')
    else:
        os.system(f'espeak -s 120 -a 180 -v en-us "{safe}" 2>/dev/null')


# ═══════════════════════════════════════════════
# SECTION 3 - IMAGE ENHANCEMENT PIPELINE
# ═══════════════════════════════════════════════

def _deskew(img: np.ndarray) -> np.ndarray:
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_image(img: np.ndarray) -> np.ndarray:
    """
    Full enhancement pipeline for maximum OCR accuracy:
      1. Upscale to min 2400px wide
      2. Denoise
      3. CLAHE contrast boost
      4. Unsharp-mask sharpening
      5. Deskew
      6. White border padding
    """
    h, w = img.shape[:2]

    # 1. Upscale
    if w < 2400:
        scale = 2400 / w
        img   = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_LANCZOS4)
        print(f"  Upscaled {w}x{h} -> {img.shape[1]}x{img.shape[0]}")

    # 2. Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                          templateWindowSize=7,
                                          searchWindowSize=21)

    # 3. CLAHE
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    img     = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Unsharp mask
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    img  = cv2.addWeighted(img, 1.65, blur, -0.65, 0)

    # 5. Deskew
    img = _deskew(img)

    # 6. Padding
    img = cv2.copyMakeBorder(img, 60, 60, 60, 60,
                             cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return img


# ═══════════════════════════════════════════════
# SECTION 4 - CAMERA
# ═══════════════════════════════════════════════

def capture_frame(device: int, width: int = 1920, height: int = 1080,
                  warmup: int = 50) -> tuple[np.ndarray, str]:
    """
    Capture a fresh frame.
    Returns (raw_cv2_frame, base64_string_for_ollama)
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        speak_text("Camera error. Please check the connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)

    frame = None
    for _ in range(warmup):
        ok, frame = cap.read()
        time.sleep(0.03)
    cap.release()

    if not ok or frame is None:
        speak_text("Failed to capture image.")
        sys.exit(1)

    _, buf  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_b64 = base64.b64encode(buf).decode('utf-8')
    return frame, img_b64


# ═══════════════════════════════════════════════
# SECTION 5 - RAPIDOCR TEXT EXTRACTION
# ═══════════════════════════════════════════════

def extract_text_ocr(frame: np.ndarray) -> str:
    """
    Enhance the frame, then run RapidOCR (ONNX) for ~95% text accuracy.
    Falls back to Moondream if RapidOCR is not installed.
    """
    if not OCR_AVAILABLE:
        return ""

    enhanced = enhance_image(frame.copy())

    # Save for RapidOCR (it accepts a file path or numpy array)
    enhanced_path = os.path.join(TEMP_DIR, "temp_enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 98])

    try:
        result, _ = _ocr(enhanced_path)
    except Exception as e:
        print(f"  RapidOCR error: {e}")
        return ""

    if not result:
        # Retry with 180-degree rotation (upside-down page)
        rotated = cv2.rotate(enhanced, cv2.ROTATE_180)
        rot_path = os.path.join(TEMP_DIR, "temp_rot.jpg")
        cv2.imwrite(rot_path, rotated)
        try:
            result, _ = _ocr(rot_path)
        except Exception:
            result = None

    if not result:
        return ""

    # Filter by confidence >= 50% and join all lines
    lines = []
    print(f"\n{'='*48}")
    print(f"  {'TEXT':<36}  CONF")
    print(f"{'='*48}")
    for item in result:
        try:
            text  = str(item[1]).strip()
            score = float(item[2])
            mark  = "[OK]" if score >= 0.50 else "[--]"
            print(f"  {mark} {text:<36}  {score:.0%}")
            if score >= 0.50 and text:
                lines.append(text)
        except (IndexError, TypeError, ValueError):
            continue
    print(f"{'='*48}\n")

    return " ".join(lines)


# ═══════════════════════════════════════════════
# SECTION 6 - MOONDREAM2 VIA OLLAMA
# ═══════════════════════════════════════════════

def ask_moondream(prompt: str, img_b64: str, timeout: int = 90) -> str:
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "moondream", "prompt": prompt,
                  "images": [img_b64], "stream": False},
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  Ollama error: {e}")
        return ""


def check_alignment(img_b64: str) -> str:
    """Returns READY, MOVE LEFT, MOVE RIGHT, MOVE UP, MOVE DOWN, ROTATE, or BLURRY."""
    prompt = (
        "Is there a document, book, or page with readable text visible and properly placed? "
        "If it needs adjustment, respond with ONLY one of: MOVE LEFT, MOVE RIGHT, MOVE UP, "
        "MOVE DOWN, ROTATE, BLURRY. If it is properly positioned say ONLY: READY"
    )
    result = ask_moondream(prompt, img_b64, timeout=30).upper()
    for kw in ["READY", "MOVE LEFT", "MOVE RIGHT", "MOVE UP", "MOVE DOWN", "ROTATE", "BLURRY"]:
        if kw in result:
            return kw
    return "READY"


def explain_with_context(img_b64: str, extracted_text: str, memory: str) -> str:
    """
    Explain using the already-extracted text as context.
    This dramatically boosts accuracy because Moondream doesn't need to re-read the text.
    """
    mem_part  = f"\n\nPrevious pages for context:\n{memory}" if memory else ""
    text_part = f'\n\nThe text on this page reads:\n"{extracted_text}"' if extracted_text else ""
    prompt    = (
        "You are helping a visually impaired person understand a book."
        + text_part
        + "\nBased on this text and the image, explain what this page is about "
          "in clear, simple language."
        + mem_part
    )
    return ask_moondream(prompt, img_b64)


def summarize_with_context(extracted_text: str, img_b64: str) -> str:
    text_part = f'\n\nThe text on this page reads:\n"{extracted_text}"' if extracted_text else ""
    prompt    = (
        "You are helping a visually impaired person."
        + text_part
        + "\nGive a short summary of this page in 1 to 2 simple sentences."
    )
    return ask_moondream(prompt, img_b64)


# ═══════════════════════════════════════════════
# SECTION 7 - PAGE MEMORY
# ═══════════════════════════════════════════════

def get_memory_context() -> str:
    if not page_memory:
        return ""
    recent = page_memory[-3:]
    return "\n".join(f"Page {p['page']}: {p['text'][:300]}" for p in recent)


def add_to_memory(page_num: int, text: str) -> None:
    global current_page
    current_page = page_num

    if len(page_memory) >= WARN_MEMORY_AT:
        speak_text(
            f"Memory is getting full with {len(page_memory)} pages stored. "
            "Should I delete the oldest pages to free memory?"
        )
        if wait_for_yn("Delete oldest pages from memory?"):
            while len(page_memory) > MAX_MEMORY_PAGES // 2:
                removed = page_memory.pop(0)
                print(f"  Flushed page {removed['page']} from memory (FIFO).")
            speak_text("Memory cleared. Old pages removed.")
        else:
            speak_text("Okay, keeping all pages in memory.")

    page_memory.append({"page": page_num, "text": text})
    print(f"  Memory: {len(page_memory)}/{MAX_MEMORY_PAGES} pages stored.")
    save_session()  # Auto-save to disk after every page


# ═══════════════════════════════════════════════
# SECTION 8 - ALIGNMENT LOOP
# ═══════════════════════════════════════════════

# ===============================================
# SECTION 8 - AUTO PAGE DETECTION (FRAME DIFF)
# ===============================================

# Motion detection settings
MOTION_THRESHOLD  = 8000   # Pixel change count to consider as motion
SETTLE_SECONDS    = 2.0    # Seconds of stillness before declaring page placed
REMOVAL_THRESHOLD = 6000   # Pixel change count to detect page removal
WATCH_FPS_SLEEP   = 0.08   # ~12 FPS watch rate (light on CPU)


def _frame_diff_score(gray1: np.ndarray, gray2: np.ndarray) -> int:
    """Return the number of pixels that changed significantly between two grayscale frames."""
    diff   = cv2.absdiff(gray1, gray2)
    _, thr = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thr)


def watch_for_page(cap: cv2.VideoCapture) -> np.ndarray:
    """
    Continuously compare frames. When the scene is stable for SETTLE_SECONDS,
    a page is considered placed. Returns the stable frame for OCR.
    """
    print("  [WATCH] Monitoring for page placement...")
    prev_gray    = None
    stable_since = None

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        score = _frame_diff_score(prev_gray, gray)
        prev_gray = gray

        if score > MOTION_THRESHOLD:
            # Something is moving - reset stability timer
            if stable_since is not None:
                stable_since = None
                print("  [WATCH] Motion detected - waiting for stillness...")
        else:
            # Scene is still
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= SETTLE_SECONDS:
                # Stable for long enough - page is placed!
                print("  [WATCH] Page stable - capturing!")
                return frame

        time.sleep(WATCH_FPS_SLEEP)


def wait_for_removal(cap: cv2.VideoCapture, reference: np.ndarray) -> None:
    """
    Wait until the current scene looks significantly different from the
    captured page frame (i.e., the page has been removed).
    """
    print("  [WATCH] Waiting for page removal...")
    ref_gray = cv2.GaussianBlur(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (21, 21), 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        gray  = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        score = _frame_diff_score(ref_gray, gray)
        if score > REMOVAL_THRESHOLD:
            print("  [WATCH] Page removed - ready for next page.")
            return
        time.sleep(WATCH_FPS_SLEEP)


def open_camera(device: int, width: int = 1920, height: int = 1080) -> cv2.VideoCapture:
    """Open and configure the camera. Stays open for the whole session."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        speak_text("Camera error. Please check the connection.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    1)
    # Flush startup frames
    for _ in range(30):
        cap.read()
        time.sleep(0.03)
    print("  Camera ready.")
    return cap


def capture_best_frame(cap: cv2.VideoCapture) -> tuple[np.ndarray, str]:
    """Read one frame from the already-open camera and encode for Ollama."""
    ok, frame = cap.read()
    if not ok:
        speak_text("Failed to capture image.")
        sys.exit(1)
    _, buf  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_b64 = base64.b64encode(buf).decode('utf-8')
    return frame, img_b64


# ═══════════════════════════════════════════════
# SECTION 9 - PROCESS ONE PAGE
# ═══════════════════════════════════════════════

def process_page(cap: cv2.VideoCapture, page_num: int, stable_frame: np.ndarray) -> bool:
    print(f"\n{'='*50}")
    print(f"  PAGE {page_num}")
    print(f"{'='*50}")
    speak_text(f"Page {page_num} detected.")

    frame   = stable_frame
    _, buf  = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    # -- Step 1: Extract Text via RapidOCR ---------------------------
    print("\n[ 1/3 ] Extracting text with RapidOCR (ONNX)...")
    speak_text("Reading the text. Please wait.")
    text = extract_text_ocr(frame)

    if not text or len(text.strip()) < 5:
        print("  RapidOCR found nothing. Falling back to Moondream...")
        speak_text("Using AI vision to read the text.")
        text = ask_moondream(
            "Extract all readable text from this image exactly as written. "
            "Output only the text, no commentary.",
            img_b64
        )

    if not text or len(text.strip()) < 5:
        speak_text("I could not detect readable text. Please adjust the page and try again.")
        return False

    print(f"\n--- Extracted Text ---\n{text}\n{'-'*22}")
    add_to_memory(page_num, text)

    # -- Step 2: Read Text Aloud ------------------------------------
    print("\n[ 2/3 ] Reading text aloud...")
    speak_text(f"Here is the text from page {page_num}.")
    speak_text(text)

    # -- Step 3: Auto Explain + Ask Summary -------------------------
    print("\n[ 3/3 ] Generating explanation...")
    speak_text("Now let me explain what this page is about.")
    context     = get_memory_context()
    explanation = explain_with_context(img_b64, text, context)
    print(f"\n--- Explanation ---\n{explanation}\n{'-'*22}")
    speak_text(explanation)

    if wait_for_yn("Would you like a short summary of this page as well?"):
        speak_text("Sure. Here is a quick summary.")
        summary = summarize_with_context(text, img_b64)
        print(f"\n--- Summary ---\n{summary}\n{'-'*22}")
        speak_text(summary)
    else:
        speak_text("Okay.")

    return True
# ===============================================
# SECTION 10 - MAIN LOOP
# ===============================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Book Reader - Auto Detection + RapidOCR + Moondream2")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    # -- Session Resume Check ----------------------------------------
    if load_session():
        speak_text(
            f"I found a previous session with {len(page_memory)} pages saved, "
            f"up to page {current_page}. Would you like to resume where you left off?"
        )
        if wait_for_yn(f"Resume from page {current_page}?"):
            speak_text(f"Resuming from page {current_page + 1}. Welcome back.")
            page_num = current_page + 1
        else:
            speak_text("Starting fresh. Deleting old session.")
            delete_session()
            page_num = 1
    else:
        page_num = 1
        speak_text(
            "Welcome to the A I Book Reader. "
            "I will automatically detect each page you place under the camera, "
            "read it, and explain it for you. "
            "Use Y and N keys to answer questions. "
            "Press spacebar at any time to pause."
        )

    # -- Open Camera Once (stays open for the whole session) ---------
    print("\nOpening camera...")
    cap = open_camera(args.device)

    # -- Continuous Auto-Detection Loop ------------------------------
    try:
        while True:
            print(f"\n{'='*50}")
            speak_text("Ready. Please place the next page under the camera.")
            print("  [WATCH] Waiting for page to be placed...")

            # Watch until a stable page is detected
            stable_frame = watch_for_page(cap)

            # Announce detection
            speak_text("Page detected. Starting now.")

            # Process the detected page
            success = process_page(cap, page_num, stable_frame)

            if success:
                page_num += 1

            # Wait for the page to be physically removed
            speak_text(
                "Done with this page. "
                "Please remove it and place the next page when ready."
            )
            wait_for_removal(cap, stable_frame)
            speak_text("Page removed.")

    except KeyboardInterrupt:
        # Ctrl+C = graceful shutdown
        print("\n\n[SHUTDOWN] Ctrl+C received.")
        if wait_for_yn("Should I delete the saved session before closing?"):
            delete_session()
            speak_text("Session deleted. Goodbye!")
        else:
            speak_text(
                f"Session saved up to page {current_page}. "
                "See you next time. Goodbye!"
            )
    finally:
        cap.release()
        print("Camera released. Reader stopped.")


if __name__ == "__main__":
    main()
