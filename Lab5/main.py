import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ex 1a - in fiecare ora
# ex 1b - 761 zile, 23 ore
# ex 1c - 1124

def ex1d():
    data = pd.read_csv('Train.csv')
    frequencies = data.iloc[:, 2].values

    N = len(frequencies)

    fourier_transform = np.fft.fft(frequencies)
    fourier_transform_abs = abs(fourier_transform / N)
    frequencies_fft = np.fft.fftfreq(N)

    plt.figure(figsize=(8, 6))
    plt.plot(frequencies_fft[:N // 2], fourier_transform_abs[:N // 2])
    plt.title('Modulus of Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()


# ex 1e - nu. lipsa de "spike-uri" notabile la alte frecvente in afara de 0
if __name__ == "__main__":
    ex1d()
