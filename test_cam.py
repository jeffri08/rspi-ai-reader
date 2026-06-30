import cv2
import sys

print("Testing camera connection...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera device 0.")
    sys.exit(1)

print("Camera opened successfully.")
for i in range(5):
    ok, frame = cap.read()
    if ok:
        print(f"Frame {i}: Read success! Shape: {frame.shape}")
    else:
        print(f"Frame {i}: Read FAILED.")

cap.release()
print("Test completed.")
