import re
import random
from collections import Counter
import numpy as np


def load_dataset():
    x_data, y_label = list(), list()
    with open('sms_spam_collection.txt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            line = line.lower()
            label, text = line.split('\t')
            text = re.split('[^A-Za-z0-9]', text)
            text = list(filter(lambda word: len(word) > 2, text))
            x_data.append(text)
            y_label.append(label)

    assert len(x_data) == len(y_label)
    n_samples = len(x_data)
    idx = list(range(n_samples))
    random.shuffle(idx)
    x_data, y_label = np.array(x_data)[idx], np.array(y_label)[idx]

    x_train, x_validation = x_data[: int(0.9 * n_samples)], x_data[int(0.9 * n_samples):]
    y_train, y_validation = y_label[: int(0.9 * n_samples)], y_label[int(0.9 * n_samples):]
    #
    words_1 = []
    for idx, label in enumerate(y_train):
        if label == 'ham':
            words_1.extend(x_train[idx])
    words_1 = [a for a, b in Counter(words_1).most_common(64)]
    #
    words_2 = []
    for idx, label in enumerate(y_train):
        if label == 'spam':
            words_2.extend(x_train[idx])
    words_2 = [a for a, b in Counter(words_2).most_common(64)]
    #
    return x_train, y_train, x_validation, y_validation, words_1 + words_2


def naive_bayes(x_train, y_train, x_validation, y_validation, x_words, c):
    assert len(x_train) == len(y_train)
    for y in y_train:
        assert y in c

    p_c = np.zeros(shape=len(c))
    p_x_c = [dict() for _ in range(len(c))]
    p_x_c_array = np.zeros(shape=(len(x_words), len(c)))

    def fit():
        for k, ck in enumerate(c):
            # 从样本中选出好瓜（是/否）的序号
            ck_idx = list(filter(lambda idx: y_train[idx] == ck, range(len(x_train))))

            # 好瓜（是/否）的先验概率
            p_c[k] = len(ck_idx) / len(x_train)

            for i, xi in enumerate(x_words):
                # 好瓜中白皮的概率
                hint = 0
                for sentence in x_train[ck_idx]:
                    hint += 1 if xi in sentence else 0
                p_x_c[k][xi] = hint / len(ck_idx)
                p_x_c_array[i, k] = hint / len(ck_idx)
    fit()

    def predict():
        prediction = np.zeros(shape=(len(x_validation), len(c)))
        for k, ck in enumerate(c):
            for i, x in enumerate(x_validation):
                evi = p_c[k]
                for word in x:
                    if word not in p_x_c[k]:
                        continue
                    p_xi_ck = p_x_c[k][word]
                    evi = evi * p_xi_ck
                prediction[i, k] = evi
        return prediction
    prediction = predict()
    spams = np.greater(prediction[:, 0], 100 * prediction[:, 1])
    found = y_validation[spams]
    accuracy = len(list(filter(lambda label: label == 'spam', found))) / len(found)
    print('found: {}, accuracy: {:.2f}%'.format(len(found), 100 * accuracy))


def main():
    x_train, y_train, x_validation, y_validation, words = load_dataset()
    naive_bayes(x_train, y_train, x_validation, y_validation, words, ['spam', 'ham'])


if __name__ == '__main__':
    main()
