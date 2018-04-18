def f(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]


def g(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3383106#3383106
    return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]


def h(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    return sorted(range(len(seq)), key=seq.__getitem__)


def main():
    pass


if __name__ == '__main__':
    main()
