#!/usr/bin/env python3
"""
Simple CLI Photo Capture Script for Raspberry Pi 5
===================================================
Captures a single photo from the USB webcam and saves it to disk.

Usage:
    python3 capture_photo.py [--device 0] [--output photo.jpg] [--width 1920] [--height 1080]
"""

import argparse
import sys
import time
import cv2

def capture_photo(device: int, output_path: str, width: int, height: int):
    print(f"Connecting to camera /dev/video{device}...")
    
    # Open camera
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device)
        
    if not cap.isOpened():
        print(f"Error: Could not open camera device {device}.")
        sys.exit(1)

    # Force MJPG and resolution
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Read 30 active frames to give the camera sensor's Auto-Exposure (AE) 
    # and Auto-White Balance (AWB) algorithms time to calibrate colors.
    print("Warming up camera sensor (auto-calibrating exposure and colors)...")
    for i in range(30):
        ok, frame = cap.read()
        time.sleep(0.03)  # simulates ~30 FPS capture rate

    if ok:
        # Save frame to disk with 100% JPEG quality
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        actual_w = frame.shape[1]
        actual_h = frame.shape[0]
        print(f"Success! High-quality photo saved to: {output_path} ({actual_w}x{actual_h})")
    else:
        print("Error: Failed to capture frame from camera.")
        sys.exit(1)

    # Release camera
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture a single photo from webcam")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--output", type=str, default="photo.jpg", help="Output filename (default photo.jpg)")
    parser.add_argument("--width", type=int, default=1920, help="Capture width (default 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Capture height (default 1080)")
    args = parser.parse_args()

    capture_photo(args.device, args.output, args.width, args.height)
