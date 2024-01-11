import matplotlib.pyplot as plt
import numpy as np


def ex1():
    def unidimensional():
        mean = 5
        variance = 16

        num_samples = 1000

        samples = np.random.normal(mean, np.sqrt(variance), num_samples)
        sample_mean = np.mean(samples)

        plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-(x - mean) ** 2 / (2 * variance))
        plt.plot(x, pdf, 'k--', linewidth=2)
        plt.title('One-Dimensional Gaussian Distribution')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')

        plt.show()

        print(f"Sample Mean: {sample_mean}")
        print(f"Specified Mean: {mean}")

    def twoDimensional():

        mean = np.array([1, 2])
        covariance_matrix = np.array([[1, 0.5], [0.5, 2]])

        num_samples = 1000

        samples = np.random.multivariate_normal(mean, covariance_matrix, num_samples)

        sample_mean_x = np.mean(samples[:, 0])
        sample_mean_y = np.mean(samples[:, 1])

        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, color='blue')

        plt.scatter(sample_mean_x, sample_mean_y, color='red', marker='x', label='Sample Mean')

        plt.scatter(mean[0], mean[1], color='green', marker='o', label='Specified Mean')

        plt.title('Two-Dimensional Gaussian Distribution')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()

        plt.show()

        print(f"Sample Mean (X): {sample_mean_x}")
        print(f"Sample Mean (Y): {sample_mean_y}")
        print(f"Specified Mean: {mean}")

    unidimensional()
    twoDimensional()


if __name__ == '__main__':
    ex1()
