import math

import numpy as np
from matplotlib import pyplot as plt


def generate_fourier_matrix(size):
    matrix = []
    for x in range(size):
        row = []
        for y in range(size):
            row.append(math.e ** (2 * np.pi * 1j * x * (y / 4)))
        matrix.append(row)
    return matrix


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


if __name__ == "__main__":
    ex1()
