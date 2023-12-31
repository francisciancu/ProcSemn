import numpy as np
from matplotlib import pyplot as plt


def ex1():
    def a(N=1000):
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
        return time_series, N

    def b():
        time_series, N = a()
        autocorr = np.correlate(time_series - np.mean(time_series), time_series - np.mean(time_series), mode='full')
        autocorr /= np.max(autocorr)

        plt.figure(figsize=(8, 6))
        plt.title('Autocorrelation of the Time Series')
        plt.plot(autocorr[N - 1:])
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.show()

    b()


if __name__ == '__main__':
    ex1()
