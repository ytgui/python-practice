import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets


def load_datasets():
    x_data = np.array(
        [(.4, -.7), (-1.5, -1), (-1.4, -.9), (-1.3, -1.2), (-1.1, -.2), (-1.2, -.4), (-.5, 1.2), (-1.5, 2.1), (1, 1),
         (1.3, .8), (1.2, .5), (.2, -2), (.5, -2.4), (.2, -2.3), (0, -2.7), (1.3, 2.1)])
    y_label = np.array([0] * 8 + [1] * 8)

    n_samples, n_features = np.shape(x_data)
    idx = list(range(n_samples))
    random.shuffle(idx)
    x_data, y_label = x_data[idx], y_label[idx]

    return x_data, y_label


def main():
    x_data, y_label = load_datasets()
    y_label_vector = np.expand_dims(y_label * 2 - 1, 1)
    n_samples, n_features = np.shape(x_data)

    session = tf.Session()

    x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    W = tf.Variable(tf.zeros(shape=[n_features, 1]))
    b = tf.Variable(tf.zeros(shape=[1]))
    y_predict = tf.matmul(x, W) + b

    loss_func = 0.5 * tf.reduce_sum(tf.square(W)) + tf.reduce_sum(tf.maximum(tf.zeros_like(y_predict), 1.0 - tf.multiply(y, y_predict)))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss_func)

    vector_dist = tf.squeeze(tf.abs(y_predict) / tf.norm(W))

    session.run(tf.global_variables_initializer())

    for i in range(200):
        loss, _ = session.run([loss_func, train_step], feed_dict={x: x_data, y: y_label_vector})
        print(loss)

    W_data, b_data = session.run([W, b])
    dist = session.run(vector_dist, feed_dict={x: x_data, y: y_label_vector})
    print(dist)

    session.close()

    plt.scatter(x_data[:, 0], x_data[:, 1], c=[('red', 'green')[i] for i in y_label])

    xx = np.arange(np.min(x_data[:, 0]), np.max(x_data[:, 0]))
    yy = -W_data[0, 0] / W_data[1, 0] * xx + b_data
    plt.plot(xx, yy)

    plt.show()


if __name__ == '__main__':
    main()
