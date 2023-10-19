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


def sinusR(t):
    return np.sin(120 * np.pi * t + np.pi / 3)


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

    time_samples = np.linspace(0, T, num_samples, endpoint=False)
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
    fq = 400
    T = 4
    num_samples = int(fq * T)

    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.stem(time_samples, sinusR(time_samples), label='a')
    plt.title('a')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def ex2b():
    fq = 800
    T = 3
    num_samples = int(fq * T)

    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.stem(time_samples, sinusR(time_samples), label='b')
    plt.title('b')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def ex2c():
    frequency = 240
    T = 1

    num_samples = int(T * frequency)

    t = np.linspace(0, T, num_samples, endpoint=False)

    sawtooth_signal = t * frequency - np.floor(t * frequency)

    plt.plot(t, sawtooth_signal)
    plt.title('Semnal "Sawtooth" cu 240 Hz')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.show()


def ex2d():
    frequency = 300
    T = 0.1

    num_samples = int(T * frequency)

    t = np.linspace(0, T, num_samples, endpoint=False)

    plt.plot(t, np.sign(sinusR(t)))
    plt.title('Semnal "Square" cu 300 Hz folosind numpy.sign')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.show()


def ex2e():
    x = 128
    y = 128

    random_signal = np.random.rand(x, y)

    plt.imshow(random_signal, cmap='gray', interpolation='nearest')
    plt.title('Semnal 2D Aleator')
    plt.show()


def ex2f():
    x = 128 * 128
    combineArray = np.concatenate((np.zeros(int(x / 2)), np.ones(int(x / 2))))
    np.random.shuffle(combineArray)
    shuffled_image = combineArray.reshape(128, 128)

    plt.imshow(shuffled_image, cmap='gray',interpolation='nearest')
    plt.title('Semnal 2D Aleator')
    plt.show()


if __name__ == "__main__":
    ex2f()

    #3.a.0.0005s
    #3.b.360,000bytes
