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


if __name__ == "__main__":
    ex1()
