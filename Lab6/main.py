import matplotlib.pyplot as plt
import numpy as np


def plot_vector(x, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x)
    plt.title(title)
    plt.xlabel('Indicele n')
    plt.ylabel('Valoare')
    plt.legend()
    plt.grid(True)
    plt.show()


def ex1_grafic_separat():
    N = 100
    x = np.random.rand(N)
    plot_vector(x, 'Vectorul initial x[n]')

    for i in range(3):
        x = x * x

        plot_vector(x, f'Iteratia {i + 1}: x ← x * x')


def ex1_acelasi_grafic():
    N = 100
    x = np.random.rand(N)
    print(x)
    plot_vector(x, 'Vectorul initial x[n]')

    x1 = x * x
    x2 = x1 * x1
    x3 = x2 * x2
    plt.figure(figsize=(8, 4))
    plt.plot(x, label='x', color='blue')
    plt.plot(x1, label='x1', color='red')
    plt.plot(x2, label='x2', color='orange')
    plt.plot(x3, label='x3', color='yellow')
    plt.xlabel('Indicele n')
    plt.ylabel('Valoare')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Oscileaza(sus-jos)
    # valorile scad/cresc in functie de valorile initiale(daca valorile initiale sunt aproape de 0
    # fiecare iteratie scade -> iteratia finala este cea mai mica)


def poly_multiply_fft(p, q):
    result_length = len(p) + len(q) - 1
    fft_length = 2 ** int(np.ceil(np.log2(result_length)))

    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    p_fft = np.fft.fft(p_coeffs, n=fft_length)
    q_fft = np.fft.fft(q_coeffs, n=fft_length)

    r_fft = np.fft.ifft(p_fft * q_fft).real

    return np.rint(r_fft).astype(int)

    # o durere de cap implementarea - nu reuseam sa iau lungimea care trebuie si imi calcula gresit
    # e mai rapid cu fft


def ex2():
    N = 5
    p_coeffs = np.random.randint(-10, 11, size=N + 1)
    q_coeffs = np.random.randint(-10, 11, size=N + 1)
    p = np.poly1d(p_coeffs)
    q = np.poly1d(q_coeffs)
    print("p(x):", p)
    print("q(x):", q)

    r_coeffs = np.convolve(p_coeffs, q_coeffs)
    r = np.poly1d(r_coeffs)
    print("r(x) (convolutie):", r)

    r_direct = np.polymul(p, q)
    print("r(x) (inmultire directa):", r_direct)

    r_fft_coeffs = poly_multiply_fft(p, q)
    r_fft = np.poly1d(r_fft_coeffs[:len(r_coeffs)])

    print("r(x) (FFT):", r_fft)


def dreptunghiulara(Nw):
    return np.ones(Nw)


def hanning(Nw):
    return np.hanning(Nw)


def afisare_sinusoida(f, A, phi, Nw):
    t = np.arange(Nw)
    sinusoida = A * np.sin(2 * np.pi * f * t / Nw + phi)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(sinusoida)
    plt.title('Sinusoida cu f = {}, A = {}, φ = {} traversând fereastra dreptunghiulară'.format(f, A, phi))
    plt.xlabel('Amostră')
    plt.ylabel('Amplitudine')

    fereastra_dreptunghiulara = dreptunghiulara(Nw)
    sinusoida_dreptunghiulara = sinusoida * fereastra_dreptunghiulara

    plt.subplot(2, 1, 2)
    plt.plot(sinusoida_dreptunghiulara)
    plt.title('Sinusoida cu fereastra dreptunghiulară aplicată')
    plt.xlabel('Amostră')
    plt.ylabel('Amplitudine')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(sinusoida)
    plt.title('Sinusoida cu f = {}, A = {}, φ = {} traversând fereastra Hanning'.format(f, A, phi))
    plt.xlabel('Amostră')
    plt.ylabel('Amplitudine')

    fereastra_hanning = hanning(Nw)
    sinusoida_hanning = sinusoida * fereastra_hanning

    plt.subplot(2, 1, 2)
    plt.plot(sinusoida_hanning)
    plt.title('Sinusoida cu fereastra Hanning aplicată')
    plt.xlabel('Amostră')
    plt.ylabel('Amplitudine')

    plt.tight_layout()
    plt.show()


def ex3():
    f = 100
    A = 1
    phi = 0
    Nw = 200

    afisare_sinusoida(f, A, phi, Nw)


if __name__ == "__main__":
    ex3()
