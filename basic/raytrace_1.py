import math
import numpy as np
import matplotlib.pyplot as plt


class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        norm = self.norm()
        return Vector(self.x / norm, self.y / norm, self.z / norm)


class Ray:
    def __init__(self, origin: Vector, direction: Vector):
        self.origin = origin
        self.direction = direction.normalize()


class Sphere:
    def __init__(self, origin: Vector, radius: float):
        self.origin = origin
        self.radius = radius

    def intersect(self, ray: Ray):
        # 球面方程：(x - xo) ^ 2 + (y - yo) ^ 2 + (z - zo) ^ 2 = R ^ 2
        # 向量形式：(l - c) ^ 2 = R ^ 2
        # 代入直线方程：l = e + dp, e 为光线原点, d 为距离, p 为方向向量
        # 二次方程：a = p ^ 2, b = 2p * (e - c), c = (e - c) ^ 2 - R ^ 2
        # 返回光线到球体的距离，返回 None 表示不相交
        a = ray.direction * ray.direction
        b = 2 * ray.direction * (ray.origin - self.origin)
        c = (ray.origin - self.origin) * (ray.origin - self.origin) - self.radius * self.radius
        delta = b * b - 4 * a * c
        if delta < 0:
            return None
        return (- b - math.sqrt(delta)) / 2 * a


class Screen:
    def __init__(self, ray_origin: Vector, screen_center: Vector, width: int, height: int):
        self._ray_origin = ray_origin
        self._screen_center = screen_center
        self._width = width
        self._height = height

        # 投影平面的法向量
        p1 = self._screen_center - self._ray_origin
        # 投影平面方程 A(x - x0) + B(y - y0) + C(z - z0) = 0, (A, B, C) = p1, (x0, y0, z0) = self._screen_center
        # 再根据 width 和 height 计算出投影面的 left, top, right, bottom

    def render(self, objects):
        pixels = np.zeros(shape=[self._height, self._width], dtype=np.float32)
        for row in range(self._height):
            for col in range(self._width):
                pass


def main():
    objects = [Sphere(origin=Vector(100, 100, 100), radius=25),
               Sphere(origin=Vector(50, 50, 50), radius=10)]

    screen = Screen(ray_origin=Vector(-100, -100, -100),
                    screen_center=Vector(0, 0, 0),
                    width=320, height=240)
    screen.render(objects)


if __name__ == '__main__':
    main()
