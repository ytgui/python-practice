import numpy as np
import matplotlib.pyplot as plt


def measure_linear(n_samples):
    # cov = 0.04
    x = 1.2 * np.ones(n_samples) + 0.2 * np.random.randn(n_samples)
    for xx in x:
        yield xx


def measure_sin(n_samples):
    Fs = 1000
    T = 1 / Fs
    t = np.linspace(0, n_samples * T, n_samples)
    x = np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(n_samples)
    for xx in x:
        yield xx


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


def run_linear():
    # enough samples
    n_samples = 500

    # 1D state space model
    F = np.array([1])
    B = np.array([0])
    H = np.array([1])

    # This matrix tells the Kalman filter how much error is
    # in each action from the time you issue the commanded voltage until it actually happens.
    Q = np.array([1e-3])

    # variable
    x_priori = np.zeros(n_samples)
    x_posteriori = np.zeros(n_samples)
    P_priori = np.zeros(n_samples)
    P_posteriori = np.zeros(n_samples)
    z = np.zeros(n_samples)
    y_hat = np.zeros(n_samples)
    S = np.zeros(n_samples)
    Kg = np.zeros(n_samples)

    # simulate sensor data
    measure = measure_sin(n_samples)
    R = np.array([0.04])

    for k in range(1, n_samples):
        # 1.predict
        x_priori[k] = F * x_posteriori[k - 1]
        P_priori[k] = F * P_posteriori[k - 1] * F.T + Q

        # 2.measure
        z[k] = next(measure)

        # 3.update
        y_hat[k] = z[k] - H * x_priori[k]
        S[k] = R + H * P_priori[k] * H.T
        Kg[k] = P_priori[k] * H.T / S[k]
        x_posteriori[k] = x_priori[k] + Kg[k] * y_hat[k]
        P_posteriori[k] = (1 - Kg[k] * H) * P_priori[k] * np.transpose(np.array([1]) - Kg[k] * H)
        P_posteriori[k] = P_posteriori[k] + Kg[k] * R * np.transpose(Kg[k])

    t = np.linspace(0, 1, n_samples)
    plt.scatter(t, z, c='red')
    plt.plot(t, x_priori, c='blue')
    plt.plot(t, x_posteriori, c='green')

    Fs = 1000
    T = 1 / Fs
    t = np.linspace(0, n_samples * T, n_samples)
    frequency_analysis(t, z, Fs, n_samples)
    frequency_analysis(t, x_posteriori, Fs, n_samples)
    plt.show()


def main():
    # 测试简单一维的卡尔曼滤波器
    # 问题：过程噪声（measure noise, and it's cov）究竟是什么东西?
    run_linear()


if __name__ == '__main__':
    main()
