import cv2

print("=== DIAGNOSING WEBCAM BACKENDS & INDEXES ===")

backends = {
    "Default": None,
    "DirectShow (CAP_DSHOW)": cv2.CAP_DSHOW,
    "Media Foundation (CAP_MSMF)": cv2.CAP_MSMF
}

# Test device indexes 0, 1, 2
for device_idx in [0, 1, 2]:
    print(f"\n--- Testing Device Index: {device_idx} ---")
    for name, backend in backends.items():
        if backend is not None:
            cap = cv2.VideoCapture(device_idx, backend)
        else:
            cap = cv2.VideoCapture(device_idx)
            
        if not cap.isOpened():
            print(f"Backend [{name}]: Could not open.")
            continue
            
        # Try to grab a frame
        ok, frame = cap.read()
        if ok:
            print(f"Backend [{name}]: SUCCESS! Frame Shape: {frame.shape}")
        else:
            print(f"Backend [{name}]: Opened, but read FAILED.")
        cap.release()
