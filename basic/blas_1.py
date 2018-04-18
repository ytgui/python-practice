import numpy as np


def solve_1():
    # 求逆解法
    # Ax = b
    a = np.array([[8, -3, 2],
                  [4, 11, -1],
                  [6, 3, 12]])
    b = np.array([20, 33, 36])
    x = np.dot(np.linalg.inv(a), b)
    print('[Inverse] step: 0, error: {}'.format(np.linalg.norm(np.dot(a, x) - b)))
    print('--------------------------------------------------')


def solve_2():
    # 朴素迭代法
    # x = cx + d, 手工推导 c/d 阵
    # 迭代矩阵 c 满足的收敛条件
    # raw equation Ax = b
    a = np.array([[8, -3, 2],
                  [4, 11, -1],
                  [6, 3, 12]])
    b = np.array([20, 33, 36])

    # transform to x = Cx + d
    c = np.array([[0 / 8, 3 / 8, -2 / 8],
                  [-4 / 11, 0 / 11, 1 / 11],
                  [-6 / 12, -3 / 12, 0 / 12]])
    d = np.array([20 / 8, 33 / 11, 36 / 12])

    # iterate solve
    x = np.array([0.0, 0.0, 0.0])
    for step in range(10):
        x = np.dot(c, x) + d
        print('[Iterate] step: {0}, error: {1:.12f}'.format(step, np.linalg.norm(np.dot(a, x) - b)))
    print('--------------------------------------------------')


def solve_3():
    # Jacobi 迭代法
    # Ax = b 等价 (D - L - U)x = b 等价 Dx = (L + U)x + b
    # 其中 D 为对角阵，且原方程 A 为对角占优矩阵（diagonally dominant）
    # 对角占优：|a_ii| >= sum(a_ij), 其中 i != j
    a = np.array([[8, -3, 2],
                  [4, 11, -1],
                  [6, 3, 12]])
    b = np.array([20, 33, 36])

    d_vector = np.diag(a)
    d_matrix = np.diag(d_vector)
    lu = d_matrix - a

    x = np.array([0.0, 0.0, 0.0])
    for step in range(10):
        x = np.divide(np.dot(lu, x) + b, d_vector)
        print('[Jacobi] step: {0}, error: {1:.12f}'.format(step, np.linalg.norm(np.dot(a, x) - b)))
    print('--------------------------------------------------')


def solve_4():
    # Gauss-Seidel 迭代法1（向量）
    # (D - L)x = Ux + b，其中 D - L 为下三角矩阵
    # 收敛速度较快，此处求逆使运算向量化（如果按照元素计算可以避免求逆）
    # Though it can be applied to any matrix with non-zero elements on the diagonals,
    # convergence is only guaranteed if the matrix is either diagonally dominant,
    # or symmetric and positive definite.
    # 虽然它可以应用于对角非零的任何矩阵，但只有矩阵是对角占优的，或对称的和正定的，才能保证收敛。
    a = np.array([[8, -3, 2],
                  [4, 11, -1],
                  [6, 3, 12]])
    b = np.array([20, 33, 36])

    dl = np.tril(a)
    u = dl - a

    x = np.array([0.0, 0.0, 0.0])
    for step in range(10):
        x = np.dot(np.linalg.inv(dl), np.dot(u, x) + b)
        print('[Gauss] step: {0}, error: {1:.12f}'.format(step, np.linalg.norm(np.dot(a, x) - b)))
    print('--------------------------------------------------')


def solve_5():
    # Gauss-Seidel 迭代法3（循环，避免求逆，公式推导）
    # 此处实现过程可以避免使用 x_next 变量，因为 x 是动态更新的，访问旧 x 再写入新 x，不会冲突
    # 因此最优的写法是 solve_6()
    a = np.array([[8, -3, 2],
                  [4, 11, -1],
                  [6, 3, 12]])
    b = np.array([20, 33, 36])

    x = np.array([0.0, 0.0, 0.0])
    n = len(x)  # h == w

    for step in range(10):
        x_next = np.zeros_like(x)
        for i in range(n):
            temp = 0.0
            for j in range(i):
                temp += a[i, j] * x_next[j]
            for j in range(i + 1, n):
                temp += a[i, j] * x[j]
            x_next[i] = (b[i] - temp) / a[i, i]
        x = x_next
        print('[Gauss] step: {0}, error: {1:.12f}'.format(step, np.linalg.norm(np.dot(a, x) - b)))
    print('--------------------------------------------------')


def solve_6():
    # Gauss-Seidel 迭代法2（循环，避免求逆，Wikipedia 伪代码）
    a = np.array([[8, -3, 2],
                  [4, 11, -1],
                  [6, 3, 12]])
    b = np.array([20, 33, 36])

    x = np.array([0.0, 0.0, 0.0])
    n = len(x)

    for step in range(10):
        for i in range(n):
            temp = 0.0
            # 此处可以向量化
            for j in range(i):
                temp += a[i, j] * x[j]
            for j in range(i + 1, n):
                temp += a[i, j] * x[j]
            x[i] = (b[i] - temp) / a[i, i]
        print('[Gauss] step: {0}, error: {1:.12f}'.format(step, np.linalg.norm(np.dot(a, x) - b)))
    print('--------------------------------------------------')


def solve_7(w):
    # SOR 方法（循环）
    # w = 1 为 GS 方法
    # w < 1 低松弛方法，解决 GS 不收敛
    # w > 1 超松弛方法，加速 GS
    # 收敛条件：A 的谱半径 < 1， 0 < w < 2
    a = np.array([[8, -3, 2],
                  [4, 11, -1],
                  [6, 3, 12]])
    b = np.array([20, 33, 36])

    x = np.array([0.0, 0.0, 0.0])
    n = len(x)

    for step in range(10):
        for i in range(n):
            temp = 0.0
            for j in range(i):
                temp += a[i, j] * x[j]
            for j in range(i + 1, n):
                temp += a[i, j] * x[j]
            x[i] = (1 - w) * x[i] + w * (b[i] - temp) / a[i, i]
        print('[SOR] step: {0}, error: {1:.12f}'.format(step, np.linalg.norm(np.dot(a, x) - b)))
    print('--------------------------------------------------')


def main():
    solve_1()
    solve_2()
    solve_3()
    solve_6()
    solve_7(1.1)


if __name__ == '__main__':
    main()
