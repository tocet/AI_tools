from smbus2 import SMBus
from time import sleep, time
from collections import deque

import matplotlib.pyplot as plt

MPU6050_ADDR = 0x68

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B

ACCEL_SCALE = 16384.0

WINDOW_SIZE = 200
SAMPLE_TIME = 0.05  # 20 Hz


def read_word_2c(bus, addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)

    value = (high << 8) + low

    if value >= 0x8000:
        value = value - 65534 #0xFFFF - 1

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
    time_data = deque(maxlen=WINDOW_SIZE)
    ax_data = deque(maxlen=WINDOW_SIZE)
    ay_data = deque(maxlen=WINDOW_SIZE)
    az_data = deque(maxlen=WINDOW_SIZE)

    plt.ion()

    fig, ax_plot = plt.subplots()
    line_x, = ax_plot.plot([], [], label="ax")
    line_y, = ax_plot.plot([], [], label="ay")
    line_z, = ax_plot.plot([], [], label="az")

    ax_plot.set_title("Dane z akcelerometru MPU6050")
    ax_plot.set_xlabel("Czas [s]")
    ax_plot.set_ylabel("Przyspieszenie [g]")
    ax_plot.set_ylim(-2, 2)
    ax_plot.grid(True)
    ax_plot.legend()

    with SMBus(1) as bus:
        bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

        print("Plot")

        start_time = time()

        try:
            while True:
                current_time = time() - start_time

                ax, ay, az = read_acceleration(bus)

                time_data.append(current_time)
                ax_data.append(ax)
                ay_data.append(ay)
                az_data.append(az)

                line_x.set_data(time_data, ax_data)
                line_y.set_data(time_data, ay_data)
                line_z.set_data(time_data, az_data)

                if len(time_data) > 2:
                    ax_plot.set_xlim(time_data[0], time_data[-1])

                fig.canvas.draw()
                fig.canvas.flush_events()

                sleep(SAMPLE_TIME)

        except KeyboardInterrupt:
            print("\nKoniec pomiaru")

        finally:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()