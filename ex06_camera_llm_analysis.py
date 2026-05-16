import cv2
import numpy as np
import requests

IMAGE_PATH = "camera_test.jpg"
OUTPUT_PATH = "camera_llm_result.jpg"

MODEL = "qwen2.5:0.5b"

REF_X = 320
REF_Y = 240
MAX_DISTANCE_PX = 40
MIN_AREA = 300
MARKER_COLOR = "red"


def get_color_mask(frame_bgr, color_name):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    if color_name == "red":
        lower1 = np.array([0, 80, 80])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 80, 80])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    elif color_name == "blue":
        lower = np.array([90, 80, 80])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif color_name == "green":
        lower = np.array([40, 80, 80])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    else:
        raise ValueError("Nieobsługiwany kolor markera.")

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detect_marker(frame_bgr):
    mask = get_color_mask(frame_bgr, MARKER_COLOR)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MIN_AREA:
        return None, mask

    moments = cv2.moments(largest)

    if moments["m00"] == 0:
        return None, mask

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    dx = cx - REF_X
    dy = cy - REF_Y
    distance = float(np.sqrt(dx**2 + dy**2))

    return {
        "detected": True,
        "center_x": cx,
        "center_y": cy,
        "area": float(area),
        "offset_x": int(dx),
        "offset_y": int(dy),
        "distance": distance
    }, mask


def rule_based_status(marker):
    if marker is None:
        return "MISSING"

    if marker["area"] < MIN_AREA:
        return "UNCERTAIN"

    if marker["distance"] <= MAX_DISTANCE_PX:
        return "OK"

    return "MISALIGNED"


def ask_local_llm(marker, status):
    if marker is None:
        measurement_text = """
marker_detected = False
marker_area = 0
center_x = None
center_y = None
offset_x = None
offset_y = None
distance_from_reference = None
"""
    else:
        measurement_text = f"""
marker_detected = True
marker_area = {marker["area"]:.1f} px^2
center_x = {marker["center_x"]} px
center_y = {marker["center_y"]} px
offset_x = {marker["offset_x"]} px
offset_y = {marker["offset_y"]} px
distance_from_reference = {marker["distance"]:.1f} px
"""

    prompt = f"""
Jesteś lokalnym asystentem diagnostycznym dla stanowiska mechatronicznego.

Kamera obserwuje element z kolorowym markerem.
Zadanie: ocenić poprawność położenia elementu względem punktu referencyjnego.

Specyfikacja techniczna:
- marker powinien być widoczny,
- pole markera powinno być większe niż {MIN_AREA} px^2,
- odległość środka markera od punktu referencyjnego powinna być mniejsza lub równa {MAX_DISTANCE_PX} px,
- jeżeli marker nie jest wykryty, stan to MISSING,
- jeżeli dane są niejednoznaczne, stan to UNCERTAIN.

Pomiary z algorytmu wizyjnego:
{measurement_text}

Klasyfikacja regułowa programu Python:
{status}

Nie zmyślaj danych.
Nie zmieniaj wartości pomiarów.
Jeżeli klasyfikacja wynika jednoznacznie z reguł, potwierdź ją.
Jeżeli widzisz ryzyko błędnej interpretacji, opisz ograniczenie.

Zwróć odpowiedź dokładnie w formacie:

STATE: OK / MISALIGNED / MISSING / UNCERTAIN
REASON:
DIAGNOSTIC_COMMENT:
RECOMMENDED_ACTION:
LIMITATIONS:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()
    return response.json()["response"]


def draw_result(frame, marker, status):
    cv2.circle(frame, (REF_X, REF_Y), MAX_DISTANCE_PX, (255, 255, 255), 2)
    cv2.circle(frame, (REF_X, REF_Y), 4, (255, 255, 255), -1)

    if marker is not None:
        cx = marker["center_x"]
        cy = marker["center_y"]

        color = (0, 255, 0) if status == "OK" else (0, 0, 255)

        cv2.circle(frame, (cx, cy), 8, color, -1)
        cv2.line(frame, (REF_X, REF_Y), (cx, cy), color, 2)

        text = (
            f"{status} | "
            f"d={marker['distance']:.1f}px | "
            f"area={marker['area']:.0f}px2"
        )
    else:
        color = (0, 0, 255)
        text = "MISSING | marker not detected"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return frame


def main():
    frame = cv2.imread(IMAGE_PATH)

    if frame is None:
        raise RuntimeError(f"Nie można odczytać obrazu: {IMAGE_PATH}")

    marker, mask = detect_marker(frame)
    status = rule_based_status(marker)

    print("--- CECHY OBRAZU ---")
    print(f"Rule-based status: {status}")

    if marker is not None:
        for key, value in marker.items():
            print(f"{key}: {value}")
    else:
        print("marker_detected: False")

    print("\n--- ANALIZA LOKALNEGO MODELU ---")
    llm_response = ask_local_llm(marker, status)
    print(llm_response)

    result = draw_result(frame.copy(), marker, status)
    cv2.imwrite(OUTPUT_PATH, result)
    cv2.imwrite("camera_mask.jpg", mask)

    print(f"\nZapisano wynik: {OUTPUT_PATH}")
    print("Zapisano maskę: camera_mask.jpg")


if __name__ == "__main__":
    main()