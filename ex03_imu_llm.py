from smbus2 import SMBus
from time import sleep, time

import numpy as np
import requests

MPU6050_ADDR = 0x68

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

ACCEL_SCALE = 16384.0

MODEL = "qwen2.5:0.5b"

SAMPLE_TIME = 0.05
MEASUREMENT_DURATION = 10.0


def read_word_2c(bus, addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)

    value = (high << 8) + low

    if value >= 0x8000:
        value = -((65535 - value) + 1)

    return value


def read_acceleration(bus):
    ax_raw = read_word_2c(bus, MPU6050_ADDR, ACCEL_XOUT_H)
    ay_raw = read_word_2c(bus, MPU6050_ADDR, ACCEL_XOUT_H + 2)
    az_raw = read_word_2c(bus, MPU6050_ADDR, ACCEL_XOUT_H + 4)

    ax = ax_raw / ACCEL_SCALE
    ay = ay_raw / ACCEL_SCALE
    az = az_raw / ACCEL_SCALE

    return ax, ay, az


def collect_measurements():
    samples = []

    with SMBus(1) as bus:
        bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

        print(f"Zbieranie danych przez {MEASUREMENT_DURATION} s...")

        start = time()

        while time() - start < MEASUREMENT_DURATION:
            ax, ay, az = read_acceleration(bus)

            magnitude = np.sqrt(ax**2 + ay**2 + az**2)

            samples.append([ax, ay, az, magnitude])

            sleep(SAMPLE_TIME)

    return np.array(samples)


def calculate_features(data):
    ax = data[:, 0]
    ay = data[:, 1]
    az = data[:, 2]
    mag = data[:, 3]

    features = {
        "samples": len(data),

        "ax_mean": float(np.mean(ax)),
        "ay_mean": float(np.mean(ay)),
        "az_mean": float(np.mean(az)),

        "ax_std": float(np.std(ax)),
        "ay_std": float(np.std(ay)),
        "az_std": float(np.std(az)),

        "magnitude_mean": float(np.mean(mag)),
        "magnitude_std": float(np.std(mag)),
        "magnitude_max": float(np.max(mag)),
        "magnitude_min": float(np.min(mag)),

        "dynamic_range": float(np.max(mag) - np.min(mag)),
        "rms_dynamic": float(np.sqrt(np.mean((mag - np.mean(mag)) ** 2))),
    }

    return features


def ask_local_llm(features):
    prompt = f"""
Jesteś lokalnym asystentem diagnostycznym dla układów mechatronicznych.

Analizujesz dane z akcelerometru MPU6050 zamontowanego na urzadzeniu.
Jednostka przyspieszenia: g.

Dane zostały zebrane przez {MEASUREMENT_DURATION} sekund.

Cechy sygnału:

liczba próbek = {features["samples"]}

średnia ax = {features["ax_mean"]:.3f}
średnia ay = {features["ay_mean"]:.3f}
średnia az = {features["az_mean"]:.3f}

odchylenie standardowe ax = {features["ax_std"]:.3f}
odchylenie standardowe ay = {features["ay_std"]:.3f}
odchylenie standardowe az = {features["az_std"]:.3f}

średnia amplituda wektora przyspieszenia = {features["magnitude_mean"]:.3f}
odchylenie standardowe amplitudy = {features["magnitude_std"]:.3f}
maksymalna amplituda = {features["magnitude_max"]:.3f}
minimalna amplituda = {features["magnitude_min"]:.3f}

zakres dynamiczny = {features["dynamic_range"]:.3f}
RMS części dynamicznej = {features["rms_dynamic"]:.3f}

Przyjmij orientacyjne kryteria:
- stan stabilny: magnitude_std < 0.05 g oraz dynamic_range < 0.20 g
- lekkie drgania: magnitude_std 0.05–0.15 g
- silne drgania lub wstrząsy: magnitude_std > 0.15 g lub magnitude_max > 1.5 g
- możliwa zmiana orientacji: średnia amplituda znacząco różna od około 1 g

Nie zmyślaj danych. Jeżeli wnioski są niepewne, napisz to.

Zwróć odpowiedź dokładnie w formacie:

STATE: STABLE / LIGHT_VIBRATION / STRONG_VIBRATION / SHOCK / UNCERTAIN
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


def main():
    data = collect_measurements()
    features = calculate_features(data)

    print("\n--- CECHY SYGNAŁU ---")
    for key, value in features.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\n--- ANALIZA LOKALNEGO MODELU ---")
    llm_response = ask_local_llm(features)
    print(llm_response)


if __name__ == "__main__":
    main()