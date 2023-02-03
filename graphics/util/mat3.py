from __future__ import annotations
from typing import Any, TypeVar
from dataclasses import dataclass
from abc import abstractmethod, ABC
import math

from graphics.util.vec3 import Pnt3, Vec3


@dataclass(frozen=True)
class Mat3(object):
    """
    A linear transformation from R^3 to R^3 in the standard basis, given by its rows
    Immutable
    """
    x0: Vec3 = Vec3(1, 0, 0)
    x1: Vec3 = Vec3(0, 1, 0)
    x2: Vec3 = Vec3(0, 0, 1)

    def __str__(self):
        return f'[[{self.x0.x}, {self.x0.y}, {self.x0.z}], [{self.x1.x}, {self.x1.y}, {self.x1.z}], [{self.x2.x}, {self.x2.y}, {self.x2.z}]]'

    def __add__(self, t: Any) -> Mat3:
        match t:
            case Mat3(x0, x1, x2):
                return Mat3(self.x0 + x0, self.x1 + x1, self.x2 + x2)
        return NotImplemented

    def __sub__(self, v: Any) -> Mat3:
        match v:
            case Mat3(x0, x1, x2):
                return Mat3(self.x0 - x0, self.x1 - x1, self.x2 - x2)
        return NotImplemented

    def __mul__(self, t: Any) -> Mat3:  # scalar multiplication / pointwise (hadamard) product
        match t:
            case int(s) | float(s):
                return Mat3(self.x0 * s, self.x1 * s, self.x2 * s)
            case Mat3(x0, x1, x2):
                return Mat3(self.x0 * x0, self.x1 * x1, self.x2 * x2)
        return NotImplemented

    def __rmul__(self, s: Any) -> Mat3:
        return self * s

    def __truediv__(self, s: Any) -> Mat3:
        match s:
            case int(s) | float(s):
                return Mat3(self.x0 / s, self.x1 / s, self.x2 / s)
        return NotImplemented

    def __neg__(self) -> Mat3:
        return Mat3(-self.x0, -self.x1, -self.x2)

    def __getitem__(self, item):
        match item:
            case (i, j) if 0 <= i <= 2 and 0 <= j <= 2:
                return self[i][j]
            case 0:
                return self.x0
            case 1:
                return self.x1
            case 2:
                return self.x2
        raise IndexError(f'index {item} out of range for {self}')

    def __matmul__(self, t: Any) -> Mat3 | Pnt3 | Vec3:
        match t:
            case Mat3():  # matrix multiplication
                T = t.transpose()
                return Mat3(Vec3(self.x0 * T.x0, self.x0 * T.x1, self.x0 * T.x2),
                            Vec3(self.x1 * T.x0, self.x1 * T.x1, self.x1 * T.x2),
                            Vec3(self.x2 * T.x0, self.x2 * T.x1, self.x2 * T.x2))
            case Pnt3():
                v = t - Pnt3(0)
                return Pnt3(self.x0 * v, self.x1 * v, self.x2 * v)
            case Vec3():
                return Vec3(self.x0 * t, self.x1 * t, self.x2 * t)
        return NotImplemented

    def transpose(self) -> Mat3:
        return Mat3(Vec3(self.x0.x, self.x1.x, self.x2.x), Vec3(self.x0.y, self.x1.y, self.x2.y), Vec3(self.x0.z, self.x1.z, self.x2.z))

    def det(self) -> float:
        return (self.x0.x * self.x1.y * self.x2.z + self.x0.y * self.x1.z * self.x2.x + self.x0.z * self.x1.x * self.x2.y -
                self.x0.x * self.x1.z * self.x2.y - self.x0.y * self.x1.x * self.x2.z - self.x0.z * self.x1.y * self.x2.x)

    def inverse(self) -> Mat3:
        return Mat3(Vec3(self.x1.y * self.x2.z - self.x1.z * self.x2.y,
                         self.x0.z * self.x2.y - self.x0.y * self.x2.z,
                         self.x0.y * self.x1.z - self.x0.z * self.x1.y),
                    Vec3(self.x1.z * self.x2.x - self.x1.x * self.x2.z,
                         self.x0.x * self.x2.z - self.x0.z * self.x2.x,
                         self.x0.z * self.x1.x - self.x0.x * self.x1.z),
                    Vec3(self.x1.x * self.x2.y - self.x1.y * self.x2.x,
                         self.x0.y * self.x2.x - self.x0.x * self.x2.y,
                         self.x0.x * self.x1.y - self.x0.y * self.x1.x)
                    ) / self.det()


@dataclass(frozen=True)
class Tform3(object):
    """
    An affine transformation from R^3 to R^3 in the standard basis
    Usually represented by a homogeneous 4x4 matrix
    Immutable
    """
    translate: Vec3 = Vec3()
    rot_scale: Mat3 = Mat3()

    def __matmul__(self, t: Any) -> Tform3 | Pnt3 | Vec3:
        match t:
            case Tform3():
                return Tform3(self.rot_scale @ t.translate + self.translate, self.rot_scale @ t.rot_scale)
            case Pnt3():
                return self.rot_scale @ t + self.translate
            case Vec3():
                return self.rot_scale @ t
        return NotImplemented

    def inverse(self) -> Tform3:
        rs_inv = self.rot_scale.inverse()
        return Tform3(rs_inv @ -self.translate, rs_inv)
