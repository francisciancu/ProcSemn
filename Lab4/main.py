import math
import numpy as np
import time
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


def ex1():
    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    my_solution_times = []
    fft_time = []
    for size in sizes:
        start_time = time.time()
        new_matrix = generate_fourier_matrix(size)
        new_matrix = []
        my_solution_times.append(round(time.time() * 1000) - round(start_time * 1000))
        start_time = time.time()
        new_matrix = np.fft.fft(np.eye(size))
        new_matrix = []
        fft_time.append(round(time.time() * 1000) - round(start_time * 1000))

    plt.figure(figsize=(12, 6))
    plt.plot(sizes, my_solution_times, label="My solution")
    plt.plot(sizes, fft_time, label="FFT solution")
    plt.title("My solution vs FFT solution")
    plt.xlabel("Sizes")
    plt.ylabel("Time")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig("ex1.png")
    plt.savefig("ex1.pdf")
    plt.show()


# ex4 - 400 - e 2 x cea mai mare frecventa (adica 200) - Nyquist

# ex7
# SNRdb = 10log10(SNR)
# 80 = 10 log10(SNR)
# log10(SNR) = 8
# SNR = 10^8
#
# Psemnaldb = 10log10(Psemnal)
# 90 = 10log10(Psemnal)
# Psemanl = 10^9
#
# SNR = Psemnal/Pzgomot
# 10^8 = 10^9/Pzgomot
# Pzgomot = 10
# Pzgomotdb = 10log10(10) = 10


if __name__ == "__main__":
    ex1()
