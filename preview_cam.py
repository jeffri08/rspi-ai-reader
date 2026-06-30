#!/usr/bin/env python3
"""
Standalone Live Preview & Capture Tool for Raspberry Pi 5
==========================================================
Displays a live webcam feed in a window on the Pi's monitor.
Allows taking photos by pressing spacebar.

Controls:
    [SPACE] - Take a photo (saved to current directory)
    [Q] or [ESC] - Quit preview

Usage:
    python3 preview_cam.py [--device 0] [--width 1280] [--height 720]
"""

import argparse
import sys
import cv2

def run_preview(device: int, width: int, height: int):
    print("Initializing camera...")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
        
    if not cap.isOpened():
        print(f"Error: Could not open camera device {device}.")
        sys.exit(1)

    # Configure camera
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Pi Cam Live Preview (Press SPACE to capture, Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n--------------------------------------------------")
    print(" STANDALONE LIVE PREVIEW RUNNING")
    print("--------------------------------------------------")
    print(" Controls:")
    print("   * Press [SPACEBAR] to take a picture")
    print("   * Press [Q] or [ESC] to quit")
    print("--------------------------------------------------\n")

    photo_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: Failed to read frame from camera.")
                break

            # Show the frame in the GUI window
            cv2.imshow(window_name, frame)

            # Wait for key press (10ms)
            key = cv2.waitKey(10) & 0xFF

            # ESC or 'q' to quit
            if key == 27 or key == ord('q'):
                print("Exiting preview...")
                break
            
            # Spacebar to take a picture
            elif key == 32:
                photo_count += 1
                filename = f"capture_{photo_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Captured and saved image as: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone camera preview and capture")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default 720)")
    args = parser.parse_args()

    run_preview(args.device, args.width, args.height)
