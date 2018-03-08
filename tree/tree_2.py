import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, tree
from collections import Counter


def tree_1():
    iris = datasets.load_iris()
    x_data, y_label = iris.data[:, [2, 3]], iris.target

    model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    model.fit(x_data, y_label)

    plot_step = 0.02
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap='RdYlBu')

    colors = [('red', 'yellow', 'blue')[yy] for yy in y_label]
    plt.scatter(x_data[:, 0], x_data[:, 1], c=colors, edgecolor='black')

    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])

    tree.export_graphviz(model, out_file='tree.dot')
    plt.show()


def tree_2():
    iris = datasets.load_iris()
    x_data, y_label = iris.data[:, [2, 3]], iris.target
    n_samples, n_features = np.shape(x_data)

    def shannon_entropy(y_sets):
        # https://www.zhihu.com/question/30828247
        ent = 0
        for key, value in Counter(y_sets).items():
            prob = value / len(y_sets)
            ent -= prob * np.log2(prob) if prob > 1e-3 else 0.0  # 0 * log 0 = 0
        return ent

    def split(y_sets, split_idx):
        yield y_sets[:split_idx]
        yield y_sets[split_idx:]
        return

    def conditional_entropy(y_sets, split_idx):
        condition_ent = 0
        for subset in split(y_sets, split_idx):
            condition_ent += (len(subset) / len(y_sets)) * shannon_entropy(subset)
        return condition_ent

    def info_gain(y_sets, split_idx):
        return shannon_entropy(y_sets) - conditional_entropy(y_sets, split_idx=split_idx)

    def get_best_split(x_sets_single_feature, y_sets):
        idx_sort = np.argsort(x_sets_single_feature)
        x_sets_sorted, y_sets_sorted = x_sets_single_feature[idx_sort], y_sets[idx_sort]

        left, right = np.min(x_sets_sorted), np.max(x_sets_sorted)
        mid = (left + right) / 2
        best_gain = info_gain(y_sets_sorted, split_idx=np.searchsorted(x_sets_sorted, mid))
        while True:
            mid_left = (left + mid) / 2
            gain_left = info_gain(y_sets_sorted, split_idx=np.searchsorted(x_sets_sorted, mid_left))

            mid_right = (mid + right) / 2
            gain_right = info_gain(y_sets_sorted, split_idx=np.searchsorted(x_sets_sorted, mid_right))

            # best_gain is optimal
            if np.greater(best_gain, max(gain_left, gain_right)):
                break

            if np.abs(gain_left - gain_right) < 1e-3:
                mid = (left + right) / 2
                break

            if np.greater(gain_left, gain_right):
                best_gain = gain_left
                right = mid
                mid = mid_left
            else:
                best_gain = gain_right
                left = mid
                mid = mid_right

            if np.equal(left, right):
                break
        return mid, best_gain  # mid is the split feature

    def build_tree(max_depth=3):
        current_x_sets, current_y_sets = x_data, y_label
        current_feature = 1
        for depth in range(max_depth + 1):
            current_feature = 1 - current_feature
            split_value, best_gain = get_best_split(current_x_sets[:, current_feature], current_y_sets)
            print('step: {0}, best_feature: {1}, split_value: {2}'.format(depth, current_feature, split_value))

            boolean_mask = np.less(current_x_sets[:, current_feature], split_value)
            left_x_sets, right_x_sets = current_x_sets[boolean_mask, :], current_x_sets[np.logical_not(boolean_mask), :]
            left_y_sets, right_y_sets = current_y_sets[boolean_mask], current_y_sets[np.logical_not(boolean_mask)]
            left_entropy, right_entropy = shannon_entropy(left_y_sets), shannon_entropy(right_y_sets)

            if left_entropy >= right_entropy:
                current_x_sets, current_y_sets = left_x_sets, left_y_sets
            else:
                current_x_sets, current_y_sets = right_x_sets, right_y_sets

    build_tree()


if __name__ == '__main__':
    # tree_1()
    tree_2()
