import numpy as np
import matplotlib.pyplot as plt


def predict_with_fixed_factor():
    estimate = 160.0
    gain_rate = 1.0
    time_step = 1.0
    g_factor, h_factor = 0.4, 0.2

    measures = np.array([158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6])
    estimations, predictions = np.zeros_like(measures), np.zeros_like(measures)
    for k, z in enumerate(measures):
        # 1.predict
        prediction = estimate + gain_rate * time_step
        predictions[k] = prediction

        # 2.update
        residual = z - prediction
        estimate = prediction + g_factor * residual
        estimations[k] = estimate
        gain_rate = gain_rate + h_factor * (residual / time_step)

    plt.scatter(range(len(measures)), measures, marker='+', label='measurements')
    plt.plot(predictions, c='blue', label='predictions')
    plt.plot(estimations, c='green', label='estimates')
    plt.legend()
    plt.show()


def main():
    predict_with_fixed_factor()


if __name__ == '__main__':
    main()
