import numpy as np
from matplotlib import pyplot as plt


def sinus(A, fq, t, faza):
    return A * np.sin((2 * np.pi * fq * t + faza) + np.pi / 2)


def cosinus(A, fq, t, faza):
    return A * np.cos(2 * np.pi * fq * t + np.pi / 2 + faza)


def ex1():
    fq = 50
    T = 1
    A = 2
    faza = np.pi / 2
    num_samples = int(fq * T)

    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    print(time_samples)
    plt.figure(figsize=(10, 6))

    plt.subplot(211)
    plt.plot(time_samples, sinus(A, fq, time_samples, faza))
    plt.title('sinus')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(time_samples, cosinus(A, fq, time_samples, faza), label='cos')
    plt.title('cosinus')
    plt.grid(True)
    plt.legend()

    plt.show()


def ex2a():
    fq = 240
    T = 1
    A = 2
    faze = [np.pi / 2, np.pi / 3, 2 * np.pi / 3, 3 * np.pi / 4]
    num_samples = int(fq * T)
    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    plt.figure(figsize=(10, 6))
    for faza in faze:
        plt.subplot(411 + faze.index(faza))
        plt.plot(time_samples, sinus(A, fq, time_samples, faza))
        plt.title('sinusoidala ' + str(faze.index(faza) + 1))
        plt.xlabel('Timp (secunde)')
        plt.ylabel('Amplitudine')
        plt.grid(True)

    plt.show()


if __name__ == "__main__":
    ex2a()
