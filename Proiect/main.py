import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops


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


def remove_watermark(watermarked_image, watermark, option="toSmaller"):

    if option in ('toSmaller', 'toBigger'):
        if option == "toSmaller":
            if watermarked_image.shape[1] > watermark.shape[1]:
                size_height = watermark.shape[1]
            else:
                size_height = watermarked_image.shape[1]

            if watermarked_image.shape[0] > watermark.shape[0]:
                size_width = watermark.shape[0]
            else:
                size_width = watermarked_image.shape[0]
        elif option == "toBigger":
            if watermarked_image.shape[1] > watermark.shape[1]:
                size_height = watermarked_image.shape[1]
            else:
                size_height = watermark.shape[1]

            if watermarked_image.shape[0] > watermark.shape[0]:
                size_width = watermarked_image.shape[0]
            else:
                size_width = watermark.shape[0]
    else:
        raise ValueError(f"Invalid parameter value: {option}. Please provide 'toSmaller' or 'toBigger'.")

    watermark_image_pil = Image.fromarray(watermarked_image)

    watermark_image_pil = watermark_image_pil.resize((size_height, size_width))

    watermark_pil = Image.fromarray(watermark)

    watermark_pil = watermark_pil.resize((size_height, size_width))

    restored_image = np.array(watermark_image_pil) - np.array(watermark_pil)

    return restored_image


def compare_origina_restored(original, watermarked, name_of_file):
    original_img = Image.fromarray(original)
    watermarked_img = Image.fromarray(watermarked)

    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')
    if watermarked_img.mode != 'RGB':
        watermarked_img = watermarked_img.convert('RGB')
    watermarked_img.save("watermarked_image_test.png")

    if original_img.mode != watermarked_img.mode:
        watermarked_img = watermarked_img.convert(original_img.mode)

    diff = ImageChops.difference(original_img, watermarked_img)

    if diff.getbbox() is None:
        return True
    else:
        diff.save(f"{name_of_file}.png")
        return False


if __name__ == "__main__":
    # host_image = datasets.face(gray=True)
    # watermark_image = datasets.face(gray=True)

    host_image = Image.open("image.jpg")
    watermark_image = Image.open("image.jpg")
    gray_host_image = host_image.convert("L")
    gray_host_image_np = np.array(gray_host_image)
    gray_watermark_image = watermark_image.convert("L")
    gray_watermark_image_np = np.array(gray_watermark_image)

    watermarked_image = embed_watermark(gray_host_image_np, gray_watermark_image_np, scaling_factor=1)

    extracted_watermark = extract_watermark(gray_watermark_image_np)

    restored_image = remove_watermark(watermarked_image, extracted_watermark)

    plt.figure(figsize=(16, 4))

    plt.subplot(141)
    plt.imshow(gray_host_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(142)
    plt.imshow(watermarked_image, cmap='gray')
    plt.title('Watermarked Image')

    plt.subplot(143)
    plt.imshow(extracted_watermark, cmap='gray')
    plt.title('Extracted Watermark')

    plt.subplot(144)
    plt.imshow(restored_image, cmap='gray')
    plt.title('Restored Image (Watermark Removed)')

    plt.savefig("image + image.png")
    plt.show()

    compare_origina_restored(gray_host_image_np, restored_image, "original vs restored_image")
    compare_origina_restored(gray_host_image_np, watermarked_image, "original vs watermarked_image")