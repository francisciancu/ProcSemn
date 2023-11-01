import math
import random

import numpy as np
from matplotlib import pyplot as plt


def generate_fourier_matrix(size):
    matrix = []
    normalization_factor = 1 / math.sqrt(size)
    for x in range(size):
        row = []
        for y in range(size):
            row.append(normalization_factor * math.e ** (2 * np.pi * 1j * x * (y / size)))
        matrix.append(row)
    return matrix


def check_matrix_is_unitary(matrix):
    matrix = np.array(matrix)
    conjugate_transpose = np.conj(np.transpose(matrix))
    product = np.dot(matrix, conjugate_transpose)
    identity = np.identity(matrix.shape[0])
    tolerance = 1e-10

    return np.allclose(product, identity, rtol=tolerance)


def ex1():
    n = 8
    fourier_matrix = generate_fourier_matrix(n)
    plt.figure(figsize=(10, 6))
    for x in range(n):
        current_row = fourier_matrix[x]
        plt.subplot(n * 100 + 10 + x + 1)
        plt.plot(np.real(current_row), label="Real")
        plt.plot(np.imag(current_row), label="Imaginary")
        plt.title(f'Row{x + 1}')
        plt.grid(True)
        plt.legend()

    plt.savefig("ex1.png")
    plt.show()

    if check_matrix_is_unitary(fourier_matrix):
        print("Matrix is unitary")
    else:
        print("Matrix is not unitary")


def ex2():
    signal_frequency = 20
    fs = 1000

    t = np.arange(0, 1, 1 / fs)

    signal = np.sin(2 * np.pi * signal_frequency * t)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(signal.real, signal.imag, c=np.abs(signal), cmap='viridis')
    plt.title("Signal Wrapping on Unit Circle")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.colorbar()
    # Nu am inteles cum trebuie facut ^ .
    # Am reusit cu internet sa fac ceva dar iese o linie dreapta si habar nu am daca e corect

    frequencies = [random.randint(1, 5), random.randint(6, 10), random.randint(11, 15), random.randint(16, 20)]
    plt.figure(figsize=(12, 6))

    for i, freq in enumerate(frequencies):
        wrapped_signal = signal * np.exp(-2j * np.pi * freq * t)
        plt.subplot(2, 2, i + 1)
        plt.plot(wrapped_signal.real, wrapped_signal.imag)
        plt.title(f'Influence of ω={freq}')
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.grid()
    # Banuiesc ca urmarim cum altereaza ω graficul?
    plt.tight_layout()
    plt.show()


def ex3():
    fs = 800
    t = np.arange(0, 2, 1 / fs)

    frequencies = [50, 200, 500]
    amplitudes = [10, 5, 2]

    signal = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)

    dft = np.fft.fft(signal)
    frequencies_dft = np.fft.fftfreq(len(dft), 1 / fs)
    magnitude_dft = np.abs(dft)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title("Signal Composed of Sinusoidal Components")

    plt.subplot(2, 1, 2)
    plt.plot(frequencies_dft, magnitude_dft)
    plt.title("Fourier Transform Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ex2()
