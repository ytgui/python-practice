import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets


def load_datasets(n_samples, n_features, n_classes):
    iris = datasets.load_iris()
    x_data, y_label = iris.data, iris.target
    x_data = (x_data - np.mean(x_data, axis=0)) / np.sqrt(np.var(x_data))

    idx = list(range(n_samples))
    random.shuffle(idx)

    sep = int(0.8 * n_samples)
    x_train, y_train, x_test, y_test = x_data[:sep], y_label[:sep], x_data[sep:], y_label[sep:]

    return x_train, y_train, x_test, y_test


def get_batch(x_data, y_label, n_batch=64):
    idx = list(range(len(x_data)))
    random.shuffle(idx)
    return x_data[idx[:n_batch]], y_label[idx[:n_batch]]


def deep_lda():
    n_samples, n_features, n_classes, n_features_target = 2000, 4, 3, 3
    x_train, y_train, x_test, y_test = load_datasets(n_samples, n_features, n_classes)

    x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    y = tf.placeholder(dtype=tf.int32, shape=[None])
    W = tf.Variable(tf.truncated_normal(shape=[n_features, n_features_target]))

    u = tf.reduce_mean(x, axis=0)

    x_cluster, u_cluster = dict(), dict()
    loss_w, loss_b = tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
    for c in range(n_classes):
        x_cluster[c] = tf.boolean_mask(x, mask=tf.equal(y, c))
        u_cluster[c] = tf.reduce_mean(x_cluster[c], axis=0)

        loss_w += tf.reduce_sum(tf.square(tf.norm(tf.matmul(x_cluster[c] - u_cluster[c], W), axis=1)))
        loss_b += tf.reduce_sum(tf.square(tf.matmul(tf.expand_dims(u_cluster[c] - u, axis=0), W)))

    loss_func = loss_w / loss_b
    train_step = tf.train.AdamOptimizer().minimize(loss_func)
    # train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss_func)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    loss_history = []
    for _ in range(2000):
        x_batch, y_batch = get_batch(x_train, y_train)
        loss, _ = session.run([loss_func, train_step], feed_dict={x: x_batch, y: y_batch})
        loss_history.append(loss)
        print(loss)

    session.close()

    plt.plot(loss_history)
    plt.title('deep lda training loss')
    plt.show()


if __name__ == '__main__':
    deep_lda()
