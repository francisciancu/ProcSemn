import numpy as np
from matplotlib import pyplot as plt


def ex1(N=1000):
    time = np.arange(N)
    trend = (time ** 2 + 7 * time + 628) / 10000
    season1 = 50 * np.sin(2 * np.pi * 0.05 * time)
    season2 = 30 * np.sin(2 * np.pi * 0.1 * time)
    seasonality = season1 + season2
    noise = np.random.normal(0, 10, N)
    time_series = trend + seasonality + noise

    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.title('Generated Time Series')
    plt.plot(time, time_series, label='Time Series')
    plt.legend()

    plt.subplot(412)
    plt.title('Trend Component')
    plt.plot(time, trend, label='Trend')
    plt.legend()

    plt.subplot(413)
    plt.title('Seasonality Component')
    plt.plot(time, seasonality, label='Seasonality')
    plt.legend()

    plt.subplot(414)
    plt.title('Small Variations (Noise)')
    plt.plot(time, noise, label='Noise')
    plt.legend()

    plt.tight_layout()
    plt.show()
    return time_series, time


def ex2(alpha=0.10):
    def calculate_exponential_average(original, alpha):
        new_ts = np.zeros_like(original)
        new_ts[0] = original[0]
        for i in range(1, len(original)):
            new_ts[i] = alpha * original[i] + (1 - alpha) * new_ts[i - 1]
        return new_ts

    original_time_series, time = ex1()
    new_time_series = calculate_exponential_average(original_time_series, alpha)
    plt.plot(time, original_time_series, label='Original Time Series')
    plt.plot(time, new_time_series, label='New Time Series')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    ex2()
