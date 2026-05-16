import cv2

IMAGE_PATH = "camera_test.jpg"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć kamery USB")

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Nie udało się pobrać obrazu")

cv2.imwrite(IMAGE_PATH, frame)

print(f"Zapisano zdjęcie: {IMAGE_PATH}")
