import numpy as np
from matplotlib import pyplot as plt


def ex1a():
    start = 0
    end = 0.03
    step = 0.0005

    time_axis = np.arange(start, end + step, step)
    print(time_axis)
    return time_axis


def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)


def y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)


def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)


def ex1b():
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(ex1a(), x(ex1a()), label='x(t)')
    plt.title('Semnal x(t)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(ex1a(), y(ex1a()), label='y(t)')
    plt.title('Semnal y(t)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(ex1a(), z(ex1a()), label='z(t)')
    plt.title('Semnal z(t)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def ex1c():
    fq = 200
    T = 1
    num_samples = int(fq * T)
    print(T)
    print(num_samples)

    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    print(x(time_samples))
    print(y(time_samples))
    print(z(time_samples))
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.stem(time_samples, x(time_samples), label='x[n]')
    plt.title('Eșantionare x[n] la 200 Hz')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.stem(time_samples, y(time_samples), label='y[n]')
    plt.title('Eșantionare y[n] la 200 Hz')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.stem(time_samples, z(time_samples), label='z[n]')
    plt.title('Eșantionare z[n] la 200 Hz')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def ex2a():

if __name__ == "__main__":
    ex1c()
