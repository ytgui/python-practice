import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from skimage import transform


def rotation2d(theta, dx, dy):
    return np.array([[np.cos(theta), -np.sin(theta), dx],
                     [np.sin(theta), np.cos(theta), dy],
                     [0, 0, 1]])


def generate_points(n_samples=100, max_rotation=0.5, max_translation=2.0, max_noise=0.25):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 9 * np.cos(t)
    y = 3 * np.sin(t)

    a = np.vstack((x, y))
    x, y = x + max_noise * np.random.rand(n_samples), y + max_noise * np.random.rand(n_samples)
    b = np.vstack((x, y))

    m = rotation2d(theta=max_rotation * np.random.rand(),
                   dx=max_translation * np.random.rand(),
                   dy=max_translation * np.random.rand())

    # make points homogeneous
    b_h = np.ones(shape=[3, n_samples])
    b_h[:2, :] = b
    b_h = np.matmul(m, b_h)
    b = b_h[:2, :]

    return a.T, b.T, m


def nearest_neighbors(src, dst):
    model = neighbors.NearestNeighbors(n_neighbors=1)

    # feed dst points to kd-tree
    model.fit(dst)

    # search nearest points of src in dst
    distances, indices = model.kneighbors(src)

    return distances, indices


def points_matcher(src, dst):
    model = transform.AffineTransform()

    # estimate rot and trans from src to dst (linear least square)
    model.estimate(src=src, dst=dst)

    return model.params


def main():
    n_samples = 50

    # b = trans * a
    a, b, m = generate_points(n_samples=n_samples)
    b_orin = b.copy()
    print('m:', m)

    # show
    plt.subplot(3, 3, 1)
    plt.scatter(a[:, 0], a[:, 1], marker='+', color='green')
    plt.scatter(b[:, 0], b[:, 1], marker='+', color='red')
    plt.xlim(-12, 12)
    plt.ylim(-12, 12)

    indices, m_estimated = None, None

    for i in range(8):
        distances, indices = nearest_neighbors(src=b, dst=a)
        indices = indices.ravel()

        # alignment
        m_estimated = points_matcher(src=b[indices], dst=a)

        # make points homogeneous
        b_h = np.ones(shape=[3, n_samples])
        b_h[:2, :] = b.T
        b_h = np.matmul(m_estimated, b_h)
        b = b_h[:2, :].T

        # print error
        print(np.mean(distances))

        # show
        plt.subplot(3, 3, i + 2)
        plt.scatter(a[:, 0], a[:, 1], marker='+', color='green')
        plt.scatter(b[:, 0], b[:, 1], marker='+', color='red')
        plt.xlim(-12, 12)
        plt.ylim(-12, 12)

    m_estimated = points_matcher(src=b_orin[indices], dst=a)
    print('m_estimated:', m_estimated)

    plt.show()


if __name__ == '__main__':
    main()
