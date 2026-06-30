# AI Book Reader for Raspberry Pi

A fully autonomous, voice-guided AI book reader built for the Raspberry Pi. The system uses a continuous camera feed to automatically detect when a page is placed, reads the text out loud, explains the content, and interactively offers summaries using offline local models.

## Features
- **Auto Page Detection:** Uses OpenCV motion detection (frame differencing) to automatically sense when you place a page and wait for it to settle. No buttons needed!
- **Hybrid OCR & Vision:** Extracts text lightning-fast using `RapidOCR` (ONNX). If text is messy or missing, it falls back to `Moondream2` AI vision to interpret the page.
- **Voice Interactive:** Fully voice-guided experience using `Piper TTS`. The system speaks all actions and asks for your input using keyboard keys (Y / N).
- **Session Memory:** Automatically saves your progress to disk. If you stop reading, it will ask to resume exactly where you left off on the next startup.
- **Contextual Understanding:** Uses a local Ollama instance (Moondream2) to generate smart, contextual explanations and summaries of the pages based on what you have read previously.
- **100% Offline:** Designed to run entirely locally on a Raspberry Pi without requiring cloud API keys or internet connection.

## Hardware Requirements
- **Raspberry Pi 5** (Recommended for AI processing speed)
- USB Webcam or Pi Camera Module
- Audio output (Headphones or Speakers)

## Software Prerequisites
Before running, you must install the required system dependencies on your Raspberry Pi:

1. **Ollama & Moondream2:**
   Install Ollama and pull the Moondream vision model:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull moondream
   ```

2. **Piper TTS:**
   Install the Piper Text-to-Speech CLI and download the `en_US-amy-medium` voice model. Ensure the `piper` command is available in your PATH.

## Installation

1. Clone this repository to your Raspberry Pi:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. Install the required Python packages:
   ```bash
   pip3 install -r requirements.txt --break-system-packages
   ```

## Usage

Start the reader script on your Raspberry Pi:
```bash
python3 ocr_reader.py --device 0
```
*(If you are using multiple cameras, you can change `--device 1`)*

**How to interact:**
- Place a page under the camera and wait 2 seconds. The system will detect it automatically.
- Press **Y** or **N** when the voice prompts you for actions (e.g. "Would you like a summary?", "Resume session?").
- Press **Spacebar** at any time to pause or resume reading.
- Press **Ctrl+C** to safely exit and save your session.

## Architecture & Logic Flow
1. **Watch Loop:** Camera watches for changes.
2. **Settle:** Waits 2 seconds of stillness.
3. **Extraction:** RapidOCR handles clear text. Moondream2 steps in if OCR fails.
4. **TTS:** Piper reads the text naturally.
5. **LLM:** Moondream2 explains the context and offers a summary.
6. **State:** Saves session memory (FIFO cache up to 8 pages) to `session.json`.

## License
MIT License
