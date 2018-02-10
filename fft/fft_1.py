import numpy as np
import matplotlib.pyplot as plt


def frequency_analysis(t, y, Fs, N):
    assert len(t) == len(y)

    frequency = np.linspace(0.0, 0.5 * Fs, N // 2)
    yf = np.fft.fft(y)

    def get_amplitude(yf):
        amplitude = np.abs(yf)[0:N // 2] * 2.0 / N
        amplitude[0] = np.real(yf[0]) / N
        return amplitude

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))
    ax1.plot(t, y)
    ax2.plot(frequency, get_amplitude(yf))


if __name__ == '__main__':
    Fs = 1000
    T = 1 / Fs
    N = 200

    # 1. source data
    t = np.linspace(0, N * T, N)
    y1 = np.sin(2 * np.pi * 10 * t) + 0.2 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)
    frequency_analysis(t, y1, Fs, N)

    # 2. filter
    RC = 0.02
    y2 = np.exp(-t / RC) / RC  # 1 / (1 + RCs)
    frequency_analysis(t, y2, Fs, N)

    # 3. after conv
    y3 = np.convolve(y1, y2, 'full')[0:N]
    frequency_analysis(t, y3, Fs, N)

    plt.show()
