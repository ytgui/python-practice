import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # param
    n_y = 256
    n_x = 256
    tau = 0.6
    n_t = 4000

    # vector
    n_v = 9
    xs = np.array([
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1,
    ])
    ys = np.array([
        -1, -1, -1,
        0,  0,  0,
        1,  1,   1,
    ])
    weights = np.array([
        1/36,   1/9,    1/36,
        1/9,    4/9,    1/9,
        1/36,   1/9,    1/36,
    ])

    # initial
    F_lattice = np.random.rand(
        n_y, n_x, n_v
    )

    # loop
    for t in tqdm(range(n_t)):
        # plot
        plt.imshow(F_lattice[:, :, 4])
        plt.pause(0.01)

        # fluid
        rho = np.sum(F_lattice, axis=-1)
        ux = np.sum(xs * F_lattice, axis=-1) / rho
        uy = np.sum(ys * F_lattice, axis=-1) / rho

        # collision
        #   - given f_eq for isothermal (constant temperature) fluid
        F_eq = np.zeros(F_lattice.shape)
        for i, (vx, vy, w) in enumerate(zip(xs, ys, weights)):
            F_eq[:, :, i] = w * rho * \
                (
                    1.0 +
                    3.0 * (vx * ux + vy * uy) +
                    4.5 * (vx * ux + vy * uy) ** 2 +
                    1.5 * (ux * ux + uy * uy) ** 2
                )
        F_lattice += (-1.0 / tau) * (F_lattice - F_eq)

        # Drift
        for i, (vx, vy) in enumerate(zip(xs, ys)):
            F_lattice[:, :, i] = np.roll(F_lattice[:, :, i], vx, axis=1)
            F_lattice[:, :, i] = np.roll(F_lattice[:, :, i], vy, axis=0)

    #
    plt.show()


if __name__ == '__main__':
    main()
