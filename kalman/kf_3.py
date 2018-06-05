import random
import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    N_STATES = 4
    N_MEASURES = 4
    N_CONTROLS = 0

    def __init__(self):
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float)

        # covariance
        # P 可以置为零，也可以通过 3sigma = 最大值 计算出 sigma^2 写入
        # R 是测量噪声协方差，表征不同传感器的相关程度
        # Q 是过程噪声协方差，
        self.Q = 1e-3 * np.eye(self.N_STATES, self.N_STATES, dtype=np.float)
        self.R = np.array([[0.25, 0, 0, 0],
                           [0, 0.25, 0, 0],
                           [0, 0, 0.25, 0],
                           [0, 0, 0, 0.25]], dtype=np.float)

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
    n_samples = 75

    # generate base samples
    t = np.linspace(0, 0.2 * np.pi, n_samples)
    vx, vy = 3 * np.sin(t) + 0.5 * np.random.rand(n_samples), 4 * np.cos(t) + 0.5 * np.random.rand(n_samples)
    x, y = np.cumsum(vx), np.cumsum(vy)

    # out line data
    idx_mutation = random.randint(int(0.25 * n_samples), int(0.75 * n_samples))
    y[idx_mutation] += 5 + 5 * np.random.rand()

    z_predict = np.zeros(shape=[n_samples, 4])
    z_optimal = np.zeros(shape=[n_samples, 4])

    kalman = KalmanFilter()
    for idx, (xx, yy, vxx, vyy) in enumerate(zip(x, y, vx, vy)):
        z_predict[idx] = kalman.predict()
        z_optimal[idx] = kalman.correct(np.array([xx, yy, vxx, vyy]))

    # show
    plt.scatter(x, y, marker='+')
    plt.plot(z_predict[:, 0], z_predict[:, 1], color='blue')
    plt.plot(z_optimal[:, 0], z_optimal[:, 1], color='green')
    plt.show()


def compute_cov():
    n_samples = 500

    # generate base samples
    t = np.linspace(0, 0.2 * np.pi, n_samples)
    vx, vy = 3 * np.sin(t) + 0.5 * np.random.rand(n_samples), 4 * np.cos(t) + 0.5 * np.random.rand(n_samples)
    x, y = np.cumsum(vx), np.cumsum(vy)

    a = np.vstack([x, y, vx, vy]).T
    b = np.cov(a, rowvar=False)
    c = 0


if __name__ == '__main__':
    # compute_cov()
    main()
