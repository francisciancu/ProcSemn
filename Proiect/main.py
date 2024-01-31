import matplotlib.pyplot as plt
import numpy as np
from scipy import datasets
from PIL import Image
import cv2


def haar_wavelet_transform(matrix):
    rows, cols = matrix.shape
    transformed_matrix = matrix.copy().astype(np.float64)

    for i in range(rows):
        transformed_matrix[i, :] = haar_wavelet_1d(transformed_matrix[i, :])

    for j in range(cols):
        transformed_matrix[:, j] = haar_wavelet_1d(transformed_matrix[:, j])

    LL = transformed_matrix[:rows // 2, :cols // 2]
    LH = transformed_matrix[:rows // 2, cols // 2:]
    HL = transformed_matrix[rows // 2:, :cols // 2]
    HH = transformed_matrix[rows // 2:, cols // 2:]

    plt.subplot(141)
    plt.imshow(LL, cmap='gray')
    plt.title('LL Quadrant')

    plt.subplot(142)
    plt.imshow(LH, cmap='gray')
    plt.title('LH Quadrant')

    plt.subplot(143)
    plt.imshow(HL, cmap='gray')
    plt.title('HL Quadrant')

    plt.subplot(144)
    plt.imshow(HH, cmap='gray')
    plt.title('HL Quadrant')

    plt.show()

    return transformed_matrix

def inverse_haar_wavelet_transform(matrix):
    rows, cols = matrix.shape
    transformed_matrix = matrix.copy().astype(np.float64)

    for i in range(rows):
        transformed_matrix[i, :] = inverse_haar_wavelet_1d(transformed_matrix[i, :])

    for j in range(cols):
        transformed_matrix[:, j] = inverse_haar_wavelet_1d(transformed_matrix[:, j])

    return transformed_matrix

def haar_wavelet_1d(signal):
    length = len(signal)
    output = np.zeros(length)

    for i in range(0, length, 2):
        average = (signal[i] + signal[i + 1]) / np.sqrt(2)
        difference = (signal[i] - signal[i + 1]) / np.sqrt(2)

        output[i // 2] = average
        output[length // 2 + i // 2] = difference

    return output

def inverse_haar_wavelet_1d(signal):
    length = len(signal)
    output = np.zeros(length)

    for i in range(0, length - 1, 2):
        average = signal[i // 2]
        difference = signal[length // 2 + i // 2]

        output[i] = (average + difference) / np.sqrt(2)
        output[i + 1] = (average - difference) / np.sqrt(2)

    return output

def embed_watermark(host_image, watermark_image, scaling_factor=0.5):
    host_transformed = haar_wavelet_transform(host_image)

    region = host_transformed[:host_transformed.shape[0] // 2, :host_transformed.shape[1] // 2]

    watermark_image_pil = Image.fromarray(watermark_image)

    watermark_image_pil = watermark_image_pil.resize((region.shape[1], region.shape[0]))

    resized_watermark = np.array(watermark_image_pil)

    region += scaling_factor * resized_watermark

    watermarked_image = inverse_haar_wavelet_transform(host_transformed)

    return watermarked_image

def extract_watermark(watermarked_image):
    watermarked_transformed = haar_wavelet_transform(watermarked_image)

    extracted_watermark = watermarked_transformed[:watermarked_image.shape[0], :watermarked_image.shape[1]]

    extracted_watermark = inverse_haar_wavelet_transform(extracted_watermark)

    return extracted_watermark

def remove_watermark(watermarked_image, watermark):
    restored_image = watermarked_image.astype(np.float64) - watermark.astype(np.float64)

    restored_image = np.clip(restored_image, 0, 255)

    return restored_image


if __name__ == "__main__":
    # host_image = datasets.face(gray=True)
    # watermark_image = datasets.face(gray=True)


    host_image = Image.open("image.jpg")
    watermark_image = Image.open("watermark2.jpg")
    gray_host_image = host_image.convert("L")
    gray_host_image_np = np.array(gray_host_image)
    gray_watermark_image = watermark_image.convert("L")
    gray_watermark_image_np = np.array(gray_watermark_image)

    watermarked_image = embed_watermark(gray_host_image_np, gray_watermark_image_np)

    extracted_watermark = extract_watermark(gray_watermark_image_np)

    # restored_image = remove_watermark(gray_watermark_image_np, extracted_watermark)

    scaled_extracted_watermark = extracted_watermark / 0.5

    plt.figure(figsize=(16, 4))

    plt.subplot(131)
    plt.imshow(gray_host_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(watermarked_image, cmap='gray')
    plt.title('Watermarked Image')

    plt.subplot(133)
    plt.imshow(scaled_extracted_watermark, cmap='gray')
    plt.title('Extracted Watermark')

    # plt.savefig("image+image.png")

    # plt.subplot(144)
    # plt.imshow(restored_image, cmap='gray')
    # plt.title('Restored Image (Watermark Removed)')

    plt.show()
