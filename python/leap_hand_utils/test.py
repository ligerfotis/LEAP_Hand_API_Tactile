import cv2

# Using the device path explicitly:
cap = cv2.VideoCapture("/dev/video10")

if not cap.isOpened():
    print("Failed to open /dev/video10")
else:
    ret, frame = cap.read()
    if ret:
        print("Frame captured successfully")
    else:
        print("Failed to capture frame from /dev/video10")
    cap.release()
