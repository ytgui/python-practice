import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    N_STATES = 4
    N_MEASURES = 2
    N_CONTROLS = 0

    def __init__(self):
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float)

        # process noise cov
        self.Q = 1e-6 * np.eye(self.N_STATES, self.N_STATES, dtype=np.float)
        self.R = np.array([[0.25, 0],
                           [0, 0.25]], dtype=np.float)

        #
        self.x_priori = np.zeros(shape=self.N_STATES, dtype=np.float)
        self.x_posteriori = np.zeros(shape=self.N_STATES, dtype=np.float)
        self.P_priori = np.zeros(shape=[self.N_STATES, self.N_STATES])
        self.P_posteriori = np.zeros(shape=[self.N_STATES, self.N_STATES])

    def predict(self):
        self.x_priori = self.F.dot(self.x_posteriori)
        self.P_priori = self.F.dot(self.P_posteriori).dot(self.F.T) + self.Q
        return self.H.dot(self.x_priori)

    def correct(self, z):
        y_hat = z - self.H.dot(self.x_priori)
        S = self.R + self.H.dot(self.P_priori).dot(self.H.T)

        # solve equation
        # K = self.P_priori.dot(self.H.T).dot(np.linalg.inv(S))
        K, residuals, rank, s = np.linalg.lstsq(S.T, self.H.dot(self.P_priori.T), rcond=-1)
        K = K.T

        self.x_posteriori = self.x_priori + K.dot(y_hat)
        self.P_posteriori = np.matmul((np.identity(self.N_STATES) - K.dot(self.H)).dot(self.P_priori),
                                      (np.identity(self.N_STATES) - K.dot(self.H)).T) + K.dot(self.R).dot(K.T)
        return self.H.dot(self.x_posteriori)


def main():
    # 高维卡尔曼跟随器（无输入）
    kf = KalmanFilter()

    n_samples = 250
    t = np.linspace(0, 2 * np.pi, n_samples)
    x, y = 1 * np.sin(t) + 3 * np.cos(t), 4 * np.sin(t) + 1 * np.cos(t)
    # x, y = 3 * t, 4 * t
    z = np.stack((x + 0.5 * np.random.randn(n_samples), y + 0.5 * np.random.randn(n_samples)), axis=-1)
    z_predict = np.zeros_like(z)
    z_optimal = np.zeros_like(z)

    for k in range(n_samples):
        z_predict[k] = kf.predict()
        z_optimal[k] = kf.correct(z[k])

    plt.plot(x, y, c='red', label='ground truth')
    plt.scatter(z[:, 0], z[:, 1], marker='+', c='red', label='measurement')
    plt.plot(z_predict[:, 0], z_predict[:, 1], c='blue', label='predict')
    plt.plot(z_optimal[:, 0], z_optimal[:, 1], c='green', label='optimal')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
