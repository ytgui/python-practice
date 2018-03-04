import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets


def load_datasets(n_samples, n_features, n_classes):
    x_data, y_label = datasets.make_classification(n_samples=n_samples,
                                                   n_features=n_features,
                                                   n_informative=(n_features - 1),
                                                   n_redundant=0,
                                                   n_repeated=1,
                                                   n_classes=n_classes)

    idx = list(range(n_samples))
    random.shuffle(idx)

    sep = int(0.8 * n_samples)
    x_train, y_train, x_test, y_test = x_data[:sep], y_label[:sep], x_data[sep:], y_label[sep:]

    return x_train, y_train, x_test, y_test


def get_batch(x_data, y_label, n_batch=64):
    idx = list(range(n_batch))
    random.shuffle(idx)
    return x_data[idx], y_label[idx]


def lda_3():
    n_samples, n_features, n_classes = 2000, 4, 3
    x_train, y_train, x_test, y_test = load_datasets(n_samples, n_features, n_classes)

    x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    y = tf.placeholder(dtype=tf.int32, shape=[None])
    W = tf.Variable(tf.truncated_normal(shape=[n_features, n_classes]))

    u = tf.reduce_mean(x, axis=0)

    xi, ui = dict(), dict()
    s_w, s_b = tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
    for i in range(n_classes):
        xi[i] = tf.boolean_mask(x, tf.equal(y, i))
        ui[i] = tf.reduce_mean(xi[i], axis=0)
        s_w += tf.matmul(tf.matmul(tf.transpose(W), tf.matmul(tf.transpose(xi[i] - ui[i]), (xi[i] - ui[i]))), W)
        s_b += tf.norm(tf.matmul(tf.expand_dims(ui[i] - u, axis=0), W))

    loss_func = s_w / s_b
    train_step = tf.train.AdamOptimizer().minimize(loss_func)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for _ in range(100):
        x_batch, y_batch = get_batch(x_train, y_train)
        loss, _ = session.run([loss_func, train_step], feed_dict={x: x_batch, y: y_batch})
        print(loss)

    session.close()


if __name__ == '__main__':
    lda_3()
