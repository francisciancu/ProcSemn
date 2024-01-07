import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy.fft import dctn, idctn

Q_jpeg = ((16, 11, 10, 16, 24, 40, 51, 61),
          (12, 12, 14, 19, 26, 28, 60, 55),
          (14, 13, 16, 24, 40, 57, 69, 56),
          (14, 17, 22, 29, 51, 87, 80, 62),
          (18, 22, 37, 56, 68, 109, 103, 77),
          (24, 35, 55, 64, 81, 104, 113, 92),
          (49, 64, 78, 87, 103, 121, 120, 101),
          (72, 92, 95, 98, 112, 100, 103, 99))


def plotImageInfo(imgInfo):
    if len(imgInfo) == 3:
        plt.imshow(imgInfo[0], cmap=imgInfo[2])
    else:
        plt.imshow(imgInfo[0])

    plt.title(imgInfo[1])


def plot2Images(img1Info, img2Info):
    plt.subplot(121)
    plotImageInfo(img1Info)

    plt.subplot(122)
    plotImageInfo(img2Info)

    plt.show()


def insertInNdarray(ndarray1, ndarray2, coords: (int, int)):
    for indexl, line in enumerate(ndarray2):
        if len(ndarray1) <= coords[0] + indexl:
            ndarray1.append([])
        for elem in line:
            ndarray1[coords[0] + indexl].append(elem)


def encodeJPEG(img):
    img1Dctn = dctn(img)
    img1Jpeg = []

    for i in range(0, len(img1Dctn), 8):
        for j in range(0, len(img1Dctn[0]), 8):
            fragmentPrep = img1Dctn[i:8 + 8 * int(i / 8), j:8 + 8 * int(j / 8)]
            adaptedQJpeg = []

            for ind1 in range(fragmentPrep.shape[0]):
                adaptedQJpeg.append([])
                for ind2 in range(fragmentPrep.shape[1]):
                    adaptedQJpeg[ind1].append(Q_jpeg[ind1][ind2])

            fragmentAdj = adaptedQJpeg * np.round(fragmentPrep / adaptedQJpeg)
            insertInNdarray(img1Jpeg, fragmentAdj, (i, j))

    for i in range(len(img1Dctn)):
        for j in range(len(img1Dctn[0])):
            img1Dctn[i, j] = img1Jpeg[i][j]

    return img1Dctn


def decodeJPEG(img):
    return idctn(img)


def ex1(img):
    img1JpegEncoded = encodeJPEG(img)

    img1JpegDecoded = decodeJPEG(img1JpegEncoded)

    plot2Images((img, 'Initial image', plt.cm.gray), (img1JpegDecoded, 'Image after JPEG compression', plt.cm.gray))


def rgbToYCrCb(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2YCrCb)


def yCrCbToRgb(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR), cv2.COLOR_BGR2RGB)


def ex2(img):
    imgycrcb = rgbToYCrCb(img)
    channels = []

    for i in range(imgycrcb.shape[2]):
        channels.append(imgycrcb[:, :, i])

    for i, channel in enumerate(channels):
        channels[i] = encodeJPEG(channel)

    for i, channel in enumerate(channels):
        channels[i] = decodeJPEG(channel)

    for i in range(imgycrcb.shape[2]):
        imgycrcb[:, :, i] = channels[i]

    plot2Images((img, 'Initial image'), (yCrCbToRgb(imgycrcb), 'Image after JPEG compression'))


def makeQJpegWithMse(quality):
    QJpegMse = []

    for elem in Q_jpeg:
        QJpegMse.append(list(elem))

    S = 60000 / quality if quality > 0 else 60000
    if quality >= 100:
        S = 1

    for indi, line in enumerate(QJpegMse):
        for indj, elem in enumerate(line):
            x = np.floor(S * elem)
            QJpegMse[indi][indj] = x if x != 0 else 1

    return QJpegMse


def encodeJpegWithMse(img, MSE):
    img1Dctn = dctn(img)
    print(f'The frequency component {np.count_nonzero(img1Dctn)}')
    img1Jpeg = []

    quality = 5

    while True:
        if quality > 100:
            raise Exception("An image with the desired MSE could not be created")
        QJpegNew = makeQJpegWithMse(quality)
        for i in range(0, len(img1Dctn), 8):
            for j in range(0, len(img1Dctn[0]), 8):
                fragmentPrep = img1Dctn[i:8 + 8 * int(i / 8), j:8 + 8 * int(j / 8)]
                fragmentAdj = QJpegNew * np.round(fragmentPrep / QJpegNew)
                insertInNdarray(img1Jpeg, fragmentAdj, (i, j))

        MseCalc = np.square(np.subtract(img, decodeJPEG(img1Jpeg))).mean()
        print(f'Quality: {quality} MSEcalc: {MseCalc}')
        if MseCalc < MSE:
            break
        quality += 5
        img1Jpeg = []

    for i in range(len(img1Dctn)):
        for j in range(len(img1Dctn[0])):
            img1Dctn[i, j] = img1Jpeg[i][j]

    print(f'The frequency component after quantization {np.count_nonzero(img1Dctn)}')

    return img1Dctn


def ex3(img, MSE):
    imgycrcb = rgbToYCrCb(img)
    channels = []
    for i in range(imgycrcb.shape[2]):
        channels.append(imgycrcb[:, :, i])

    for i, channel in enumerate(channels):
        channels[i] = encodeJpegWithMse(channel, MSE)

    for i, channel in enumerate(channels):
        channels[i] = decodeJPEG(channel)

    for i in range(imgycrcb.shape[2]):
        imgycrcb[:, :, i] = channels[i]

    plot2Images((img, 'Initial image'),
                (yCrCbToRgb(imgycrcb), 'JPEG compression user-imposed MSE'))


if __name__ == '__main__':
    repeat = True
    while repeat:
        chooseEx = int(input("Choose what ex to run: (1-3):\n"))
        if chooseEx == 1:
            ex1(misc.ascent())
        elif chooseEx == 2:
            ex2(misc.face())
        elif chooseEx == 3:
            mseInput = input("Choose MSE value (default 300): ")
            mseValue = int(mseInput) if mseInput else 300
            ex3(misc.face(), mseValue)
        else:
            print("Not a valid choice!")
        if input("Do you want to run again? (y/n)\n") != 'y':
            repeat = False
