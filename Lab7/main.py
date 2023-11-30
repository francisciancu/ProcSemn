import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from scipy import misc


def ex1():
    def a():
        img = misc.face(gray=True)
        img_modified = []
        for i in range(len(img)):
            img_modified.append([])
            for j in range(len(img[0])):
                img_modified[i].append(np.sin(2 * np.pi * i + 3 * np.pi * j))

        plt.imshow(img_modified, cmap=plt.cm.gray)
        plt.show()

        plt.imshow(20 * np.log10(abs(np.fft.fft2(img_modified))))
        plt.colorbar()
        plt.show()

    def b():
        img = misc.face(gray=True)
        img_modified = []
        for i in range(len(img)):
            img_modified.append([])
            for j in range(len(img[0])):
                img_modified[i].append(np.sin(4 * np.pi * i) + np.sin(6 * np.pi * j))

        plt.imshow(img_modified, cmap=plt.cm.gray)
        plt.show()

        plt.imshow(20 * np.log10(abs(np.fft.fft2(img_modified))))
        plt.colorbar()
        plt.show()

    def c():
        img = misc.face(gray=True)
        imgfft = np.fft.fft2(img)

        imgfft_cutoff = imgfft.copy()

        for i in range(len(imgfft_cutoff)):
            for j in range(len(imgfft_cutoff[0])):
                if i == 0 and (j == 5 or j == len(imgfft_cutoff[0]) - 5):
                    imgfft_cutoff[i][j] = 1
                else:
                    imgfft_cutoff[i][j] = 0

        X_cutoff = np.fft.ifft2(imgfft_cutoff)
        X_cutoff = np.real(X_cutoff)
        plt.imshow(X_cutoff, cmap=plt.cm.gray)
        plt.show()

        plt.imshow(20 * np.log10(abs(imgfft)))
        plt.colorbar()
        plt.show()

    def d():
        img = misc.face(gray=True)
        imgfft = np.fft.fft2(img)

        imgfft_cutoff = imgfft.copy()

        for i in range(len(imgfft_cutoff)):
            for j in range(len(imgfft_cutoff[0])):
                if j == 0 and (i == 5 or i == len(imgfft_cutoff[0]) - 5):
                    imgfft_cutoff[i][j] = 1
                else:
                    imgfft_cutoff[i][j] = 0

        X_cutoff = np.fft.ifft2(imgfft_cutoff)
        X_cutoff = np.real(X_cutoff)
        plt.imshow(X_cutoff, cmap=plt.cm.gray)
        plt.show()

        plt.imshow(20 * np.log10(abs(imgfft)))
        plt.colorbar()
        plt.show()

    def e():
        img = misc.face(gray=True)
        imgfft = np.fft.fft2(img)

        imgfft_cutoff = imgfft.copy()

        for i in range(len(imgfft_cutoff)):
            for j in range(len(imgfft_cutoff[0])):
                if j == i == 5 or j == i == len(imgfft_cutoff[0]) - 5:
                    imgfft_cutoff[i][j] = 1
                else:
                    imgfft_cutoff[i][j] = 0

        X_cutoff = np.fft.ifft2(imgfft_cutoff)
        X_cutoff = np.real(X_cutoff)

        plt.imshow(X_cutoff, cmap=plt.cm.gray)
        plt.show()

        plt.imshow(20 * np.log10(abs(imgfft)))
        plt.colorbar()
        plt.show()

    a()
    b()
    c()
    d()
    e()
    return


def ex2(SNR=5):
    def calcSNR(img: np.ndarray, img_compressed: np.ndarray) -> float:
        snr = 0
        entries = 0

        for i in range(len(img)):
            for j in range(len(img[0])):
                currentPixImg = img[i, j]
                currentPixImgCompr = img_compressed[i, j] if img_compressed[i, j] != 0 else 0.1
                divided = currentPixImg / currentPixImgCompr
                snr += divided
                entries += 1
        return abs(snr / entries)

    cut_off = 160
    X = misc.face(gray=True)
    while True:
        print(cut_off)
        Y = np.fft.fft2(X)

        freq_db = 20 * np.log10(abs(Y))
        m_freq = freq_db.max()
        Y[freq_db > (m_freq - cut_off)] = 0

        X_cut = np.real(np.fft.ifft2(Y))
        SNRcalc = abs(calcSNR(X, X_cut))
        print((calcSNR(X, X_cut)))

        if SNRcalc < SNR:
            break
        elif cut_off < 0:
            X_cut = X
            break
        else:
            cut_off -= 20
            if cut_off == 60:
                cut_off -= 1

    plt.subplot(121).imshow(X, cmap=plt.cm.gray)
    plt.title('Original')

    plt.subplot(122).imshow(X_cut, cmap=plt.cm.gray)
    plt.title('Noisy')
    plt.show()
    return


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)
#https://stackoverflow.com/questions/25524192/how-to-get-the-signal-to-noise-ratio-from-an-image-in-python


def ex3():
    X = misc.face(gray=True)
    noise = np.random.normal(0, 0.1, X.shape)
    noisy_X = X + noise

    plt.subplot(121).imshow(X, cmap=plt.cm.gray)
    plt.title('Original')
    plt.subplot(122).imshow(noisy_X, cmap=plt.cm.gray)
    plt.title('Noisy')
    plt.show()
    original_snr = signaltonoise(X, axis=None)
    print('Original SNR:', original_snr)

    snr_noise = signaltonoise(noisy_X, axis=None)
    print('SNR of noisy:', snr_noise)

    estimated_noise = noisy_X - X
    new_X = noisy_X - estimated_noise
    snr_after = signaltonoise(new_X, axis=None)
    print('Final SNR:', snr_after)
    plt.subplot(131).imshow(noisy_X, cmap=plt.cm.gray)
    plt.title('Noisy')
    plt.subplot(132).imshow(new_X, cmap=plt.cm.gray)
    plt.title('New Original')
    plt.subplot(133).imshow(X, cmap=plt.cm.gray)
    plt.title('Original')
    plt.show()


if __name__ == "__main__":
    ex3()
