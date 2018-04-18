import random


# ----------------------------------------------
# 简单跳台阶问题
# 每次只能跳 1 或 2 级，总台阶数为 n
# 求不同跳法数目
# ----------------------------------------------
def jump_floor_iter(n):
    if n == 0:
        return 0
    if n == 1:
        return 1

    a, b = 1, 2
    for i in range(2, n):
        c = a + b
        a, b = b, c
    return b


def jump_floor_dp(n):
    if n == 0:
        return 0

    def jump(n):
        if n < 0:
            return 0
        if n in (0, 1):
            return 1
        return jump(n - 1) + jump(n - 2)

    return jump(n)


def test_jump_floor():
    for _ in range(100):
        n = random.randrange(25)
        assert jump_floor_iter(n) == jump_floor_dp(n)
    print('all tests passed')


# ----------------------------------------------
# 小偷偷房子问题
# 不能偷连续的两个房子
# 求最大收益
# ----------------------------------------------
def house_robber(nums):
    # f(k) 表示前i个房子中,偷到的最大价值，最终要解得 f(n)
    # f(0) = nums[0]
    # f(1) = max(num[0], num[1])
    # f(k) = max( f(k-2) + nums[k], f(k-1) )
    last, now = 0, 0
    for n in nums:
        last, now = now, max(last + n, now)
    return now


def test_house_robber():
    profit = house_robber([8, 4, 8, 5, 9, 6, 5, 4, 4, 10])
    print(profit)


# ----------------------------------------------
# 最大收益
# 给定一个不同时间的价格序列，求出最优买入卖出的最大收益
# ----------------------------------------------
def sell_stock(prices):
    begin, max_profit = 0, 0
    for end in range(1, len(prices)):
        profit = prices[end] - prices[begin]
        if profit <= 0:
            begin = end
        max_profit = max(max_profit, profit)
    return max_profit


def test_sell_stock():
    assert sell_stock([1]) == 0
    assert sell_stock([7, 1, 5, 3, 6, 4]) == 5
    assert sell_stock([1, 5, 3, 9, 4, 6, 2, 8]) == 8


# ----------------------------------------------
# 最大子序列
# 给定一个序列，找到其中序列元素相加最大的子序列
# ----------------------------------------------
def max_subarray(nums):
    # f(k) = max{ nums[k], f(k-1) + nums[k] }
    max_total, max_current_step = nums[0], nums[0]
    for num in nums[1:]:
        max_current_step = max(num, max_current_step + num)
        max_total = max(max_total, max_current_step)
    return max_total


def test_max_subarray():
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray([1]) == 1
    assert max_subarray([1, 2, 3, 4, -1]) == 10


# ----------------------------------------------
# 最小代价爬楼梯
# 每次可以走 1 或 2 级台阶，第一次可以从位置 0 或 1 开始
# ----------------------------------------------
def min_cost_floor(costs, idx_current):
    pass


if __name__ == '__main__':
    # 动态规划：
    # 每个阶段的最优状态，可以从之前的某一个或几个阶段的状态中得到
    min_cost_floor([1, 100, 1, 1, 1, 100, 1, 1, 100, 1])
