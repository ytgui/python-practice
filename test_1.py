from collections import Counter


a = [1, 2]
b = [0, 0]
map(lambda x: b.append(x), a)
print(b)
