def generate():
    for i in range(10):
        yield i


if __name__ == '__main__':
    gen = generate()
    x = next(gen)
    print(x)
    x = next(gen)
    print(x)
    x = next(gen)
    print(x)