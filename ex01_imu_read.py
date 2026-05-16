from smbus2 import SMBus
from time import sleep

MPU6050_ADDR = 0x68

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

ACCEL_SCALE = 16384.0  # dla zakresu ±2g


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


def main():
    with SMBus(1) as bus:
        bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

        print("Odczyt danych z akcelerometru MPU6050")
        print("Jednostka: g")
        print("Ctrl+C kończy program\n")

        try:
            while True:
                ax, ay, az = read_acceleration(bus)

                print(
                    f"ax = {ax:+.3f} g | "
                    f"ay = {ay:+.3f} g | "
                    f"az = {az:+.3f} g"
                )

                sleep(0.1)

        except KeyboardInterrupt:
            print("\nKoniec pomiaru.")


if __name__ == "__main__":
    main()