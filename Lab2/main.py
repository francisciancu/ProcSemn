import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt
from scipy.io import wavfile


def sinus(A, fq, t, faza):
    return A * np.sin((2 * np.pi * fq * t + faza) + np.pi / 2)


def sinusR(t):
    return np.sin(120 * np.pi * t + np.pi / 3)


def cosinus(A, fq, t, faza):
    return A * np.cos(2 * np.pi * fq * t + np.pi / 2 + faza)


def generate_gaussian_noise(T, num_samples, snr):
    signal_power = 0.5
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), int(num_samples * T))
    return noise


def add_noise_to_signal(signal, noise):
    return signal + noise


def choose_method(exNumber):
    if exNumber == "2a":
        fq = 400
        T = 4
        num_samples = int(fq * T)
        time_samples = np.linspace(0, T, num_samples, endpoint=False)
        return sinusR(time_samples)
    elif exNumber == "2b":
        fq = 800
        T = 3
        num_samples = int(fq * T)
        time_samples = np.linspace(0, T, num_samples, endpoint=False)
        return sinusR(time_samples)
    elif exNumber == "2c":
        frequency = 240
        T = 1
        num_samples = int(T * frequency)
        t = np.linspace(0, T, num_samples, endpoint=False)
        sawtooth_signal = t * frequency - np.floor(t * frequency)
        return sawtooth_signal
    elif exNumber == "2d":
        frequency = 300
        T = 0.1
        num_samples = int(T * frequency)
        t = np.linspace(0, T, num_samples, endpoint=False)
        return np.sign(sinusR(t))
    else:
        print("Invalid")


def ex1():
    fq = 50
    T = 1
    A = 2
    faza = np.pi / 2
    num_samples = int(fq * T)

    time_samples = np.linspace(0, T, num_samples, endpoint=False)
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
    fq = 250
    T = 1
    A = 2
    faze = [np.pi / 2, np.pi / 3, 2 * np.pi / 3, 3 * np.pi / 4]
    num_samples = 100
    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    plt.figure(figsize=(10, 6))
    for i, faza in enumerate(faze):
        plt.plot(time_samples, sinus(A, fq, time_samples, faza), label=f'Faza {i}')

    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.title('Semnale sinusoidale cu faze diferite')
    plt.legend()
    plt.grid(True)
    plt.show()


def ex2b():
    SNR = [0.1, 1, 10, 100]
    fq = 250
    T = 1
    A = 2
    faze = np.pi / 2
    num_samples = 100
    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    plt.figure(figsize=(10, 6))
    signal = sinus(A, fq, time_samples, faze)
    plt.plot(time_samples, signal, label='Semnal Original', linewidth=2)

    for snr in SNR:
        noise = generate_gaussian_noise(T, num_samples, snr)
        noisy_signal = add_noise_to_signal(signal, noise)
        plt.plot(time_samples, noisy_signal, label=f'Noisy Signal (SNR {snr} dB)')

    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.title('Noice')
    plt.legend()
    plt.grid(True)
    plt.show()


def ex3():

    # am comentat astea pentru ca am mai multe difuzoare si aparent mie imi mergea dar nu auzeam nimic
    # pentru ca folosea un device care la mine e setat pe mute

    # sd.play(choose_method("2a"), 44111, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    # sd.wait()
    # sd.play(choose_method("2b"), 44111, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    # sd.wait()
    # sd.play(choose_method("2c"), 44111, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    # sd.wait()
    # sd.play(choose_method("2d"), 44111, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    # sd.wait()

    sd.play(choose_method("2a"), 44111)
    sd.wait()
    sd.play(choose_method("2b"), 44111)
    sd.wait()
    sd.play(choose_method("2c"), 44111)
    sd.wait()
    sd.play(choose_method("2d"), 44111)
    sd.wait()

    wavfile.write("2a.wav", 44111, choose_method("2a"))

    rate, read_signal = wavfile.read("2a.wav")
    # sd.play(read_signal, rate, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    sd.play(read_signal, rate)
    sd.wait()


def ex4():
    fq_sin = 240
    fq_saw = 100
    T = 1
    A = 2
    faza = np.pi / 2
    num_samples = 200

    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    plt.figure(figsize=(10, 6))
    sinus_signal = sinus(A, fq_sin, time_samples, faza)
    sawtooth_signal = A * (time_samples * fq_saw - np.floor(time_samples * fq_saw))

    plt.subplot(311)
    plt.plot(time_samples, sinus_signal, label='Sinus')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.legend()

    plt.subplot(312)
    plt.plot(time_samples, sawtooth_signal, label='Sawtooth')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.legend()

    plt.subplot(313)
    signal_sum = sinus_signal + sawtooth_signal
    plt.plot(time_samples, signal_sum, label='Sinus+Sawtooth')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.legend()

    plt.show()


def ex5():
    fq1 = 220
    fq2 = 440
    T = 1
    A = 2
    num_samples = 100
    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    signal1 = A * np.sin(2 * np.pi * fq1 * time_samples)
    signal2 = A * np.sin(2 * np.pi * fq2 * time_samples)

    combined_signal = np.concatenate((signal1, signal2))

    # sd.play(signal1, 44100, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    sd.play(signal1, 44100)
    sd.wait()
    # sd.play(signal2, 44100, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    sd.play(signal2, 44100)
    sd.wait()
    # sd.play(combined_signal, 44100, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    sd.play(combined_signal, 44100)
    sd.wait()
    print("sunteul combinat e mai ?inalt? decat cele 2 separat")


def ex6():
    fs = 44100
    T = 10
    num_samples = 1000
    A = 2
    time_samples = np.linspace(0, T, num_samples, endpoint=False)
    fq_a = fs / 2
    signal_a = A * np.sin(2 * np.pi * fq_a * time_samples)
    fq_b = fs / 4
    signal_b = A * np.sin(2 * np.pi * fq_b * time_samples)
    fq_c = 0
    signal_c = A * np.sin(2 * np.pi * fq_c * time_samples)

    # sd.play(signal_a, fs, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    sd.play(signal_a, fs)
    sd.wait()
    # sd.play(signal_b, fs, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    sd.play(signal_b, fs)
    sd.wait()
    # sd.play(signal_c, fs, device='Speakers (HyperX QuadCast S) Windows DirectSound')
    sd.play(signal_c, fs)
    sd.wait()
    print("c si a nu se aud")
    print("c defapt nu are sunte deloc pentru ca e frecventa 0?")


def ex7():
    fs_start = 1000
    T = 1
    A = 2
    num_samples_start = int(fs_start * T)
    time_samples_start = np.linspace(0, T, num_samples_start, endpoint=False)
    signal_start = A * np.sin(2 * np.pi * fs_start * time_samples_start)

    signal_finish_a = signal_start[::4]
    time_samples_finish_a = time_samples_start[::4]

    plt.figure(figsize=(10, 6))

    plt.subplot(311)
    plt.plot(time_samples_start, signal_start, label='Semnal inițial')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.legend()

    plt.subplot(312)
    plt.plot(time_samples_finish_a, signal_finish_a, label='Semnal decimat')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.legend()

    print(" semnalul decimat nu este la fel de 'inghesuit' in grafic pentru ca se ia doar 1 din 4 valori ")
    plt.subplot(313)
    signal_finish_b = signal_start[1::4]
    time_samples_finish_b = time_samples_start[1::4]
    plt.plot(time_samples_finish_b, signal_finish_b, label='Semnal decimat (de la al doilea element)')
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Amplitudine')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("arata la fel ca cel decimat de la primul element, dar presupun (pare ca asa e din grafic) incepe de la "
          "alta valoare ")


def ex8():
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 100)
    actual_sin = np.sin(alpha)
    linear_approximation = alpha
    linear_error = np.abs(actual_sin - linear_approximation)
    pade_approximation = (alpha - ((7 * alpha ** 3) / 60)) / (1 + (alpha ** 2 / 20))

    plt.figure(figsize=(10, 6))

    plt.subplot(311)
    plt.plot(alpha, actual_sin, label='sin(α)')
    plt.plot(alpha, linear_approximation, label='Aproximare lineară (α)')
    plt.title('Aproximarea sin(α) ≈ α')
    plt.xlabel('α')
    plt.ylabel('Valoare')
    plt.legend()

    plt.subplot(312)
    plt.semilogy(alpha, linear_error, label='Eroare lineară')
    plt.title('Eroarea dintre sin(α) și Aproximarea lineară')
    plt.xlabel('α')
    plt.ylabel('Eroare')
    plt.legend()

    plt.subplot(313)
    plt.plot(alpha, actual_sin, label='sin(α)')
    plt.plot(alpha, pade_approximation, label='Aproximare Pade')
    plt.title('Aproximarea Pade sin(α)')
    plt.xlabel('α')
    plt.ylabel('Valoare')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("cu aproximarea eroarea creste pentru valori mai mari")
    print("aproximarea pade este mult mai precisa?")


if __name__ == "__main__":
    ex8()
