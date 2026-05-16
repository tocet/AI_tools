import cv2
import numpy as np

IMAGE_PATH = "camera_test.jpg"
OUTPUT_PATH = "marker_result.jpg"

# Punkt referencyjny
REF_X = 320
REF_Y = 240

# Dopuszczalna odległość markera od punktu referencyjnego
MAX_DISTANCE_PX = 40

# Minimalne pole markera
MIN_AREA = 300

# Kolor markera: red / blue / green
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
        raise ValueError("Nieobsługiwany kolor markera")

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

    return {
        "center_x": cx,
        "center_y": cy,
        "area": float(area)
    }, mask


def classify_marker(marker):
    if marker is None:
        return {
            "status": "BRAK",
            "reason": "Marker nie został wykryty"
        }

    dx = marker["center_x"] - REF_X
    dy = marker["center_y"] - REF_Y

    distance = float(np.sqrt(dx**2 + dy**2))

    if distance <= MAX_DISTANCE_PX:
        status = "OK"
        reason = "Marker znajduje się w dopuszczalnym obszarze"
    else:
        status = "MISALIGNED"
        reason = "Marker znajduje się poza dopuszczalnym obszarem"

    marker["offset_x"] = dx
    marker["offset_y"] = dy
    marker["distance"] = distance

    return {
        "status": status,
        "reason": reason
    }


def draw_result(frame, marker, classification):
    cv2.circle(frame, (REF_X, REF_Y), MAX_DISTANCE_PX, (255, 255, 255), 2)
    cv2.circle(frame, (REF_X, REF_Y), 4, (255, 255, 255), -1)

    if marker is not None:
        cx = marker["center_x"]
        cy = marker["center_y"]

        color = (0, 255, 0) if classification["status"] == "OK" else (0, 0, 255)

        cv2.circle(frame, (cx, cy), 8, color, -1)
        cv2.line(frame, (REF_X, REF_Y), (cx, cy), color, 2)

        text = (
            f"{classification['status']} | "
            f"dx={marker['offset_x']} px, "
            f"dy={marker['offset_y']} px, "
            f"d={marker['distance']:.1f} px"
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
    classification = classify_marker(marker)

    result = draw_result(frame.copy(), marker, classification)

    print("--- WYNIK DETEKCJI ---")
    print(f"Status: {classification['status']}")
    print(f"Powód: {classification['reason']}")

    if marker is not None:
        print(f"Center x: {marker['center_x']}")
        print(f"Center y: {marker['center_y']}")
        print(f"Area: {marker['area']:.1f}")
        print(f"Offset x: {marker['offset_x']}")
        print(f"Offset y: {marker['offset_y']}")
        print(f"Distance: {marker['distance']:.1f}")

    cv2.imwrite(OUTPUT_PATH, result)
    cv2.imwrite("marker_mask.jpg", mask)

    print(f"Zapisano wynik: {OUTPUT_PATH}")
    print("Zapisano maskę: marker_mask.jpg")


if __name__ == "__main__":
    main()