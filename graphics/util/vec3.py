from __future__ import annotations
from typing import Any, TypeVar, Tuple
from dataclasses import dataclass
from abc import abstractmethod, ABC
import math

from potto import Const, Sqrt

@dataclass(frozen=True)
class AbstractVector3(ABC):
    x = Const(0)
    y = Const(0)
    z = Const(0)

    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __getitem__(self, item):
        match item:
            case 0:
                return self.x
            case 1:
                return self.y
            case 2:
                return self.z
        raise IndexError(f'index {item} out of range for {self}')

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(frozen=True)
class Pnt3(AbstractVector3):
    """
    Points of a (3d) Euclidean affine plane
    Immutable

    Pnt2(0, 0) is the origin
    """
    x: float = 0.
    y: float = 0.
    z: float = 0.

    def __add__(self, v: Any) -> Pnt3:
        match v:
            case Pnt3():
                return NotImplemented  # NOTE: We disallow adding Points to Points, that's the main difference with Vec
            case Vec3(x, y, z):
                return Pnt3(self.x + x, self.y + y, self.z + z)
        return NotImplemented

    def __sub__(self, v: Any) -> Pnt3 | Vec3:
        match v:
            case Pnt3(x, y, z):
                return Vec3(self.x - x, self.y - y, self.z - z)  # Pnt - Pnt => Vec
            case Vec3(x, y, z):
                return Pnt3(self.x - x, self.y - y, self.z - z)  # Pnt - Vec => Pnt
        return NotImplemented

    def __mul__(self, s: Any) -> Pnt3:
        match s:
            case int(s) | float(s):
                return Pnt3(self.x * s, self.y * s, self.z * s)  # identifying [point * s] with [origin + (point - origin)*s]
        return NotImplemented

    def __rmul__(self, s: Any) -> Pnt3:
        match s:
            case int(s) | float(s):
                return self * s
        return NotImplemented

    def __truediv__(self, s: Any) -> Pnt3:
        match s:
            case int(s) | float(s):
                return Pnt3(self.x / s, self.y / s, self.z / s)
        return NotImplemented

    def __neg__(self) -> Pnt3:
        return Pnt3(-self.x, -self.y, -self.z)


@dataclass(frozen=True)
class Vec3(AbstractVector3):
    """
    Vectors from a (3d) real Euclidean vector space equipped with inner product and induced metric
    Immutable

    Secretly we can also use math.inf for extended reals
    """
    x: float = 0.
    y: float = 0.
    z: float = 0.

    def __add__(self, v: Any) -> Pnt3 | Vec3:
        match v:
            case Pnt3(x, y, z):
                return Pnt3(self.x + x, self.y + y, self.z + z)
            case Vec3(x, y, z):
                return Vec3(self.x + x, self.y + y, self.z + z)
        return NotImplemented

    def __sub__(self, v: Any) -> Vec3:
        match v:
            case Vec3(x, y, z):
                return Vec3(self.x - x, self.y - y, self.z - z)
        return NotImplemented

    def __mul__(self, s: Any) -> Vec3 | float:
        match s:
            case int(s) | float(s):
                return Vec3(self.x * s, self.y * s, self.z * s)
            case Vec3(x, y, z):  # overload for dot product
                return self.x * x + self.y * y + self.z * z
        return NotImplemented

    def __rmul__(self, s: Any) -> Vec3:
        match s:
            case int(s) | float(s):
                return self * s
            case Vec3():
                return self * s
        return NotImplemented

    def __truediv__(self, s: Any) -> Vec3:
        match s:
            case int(s) | float(s):
                return Vec3(self.x / s, self.y / s, self.z / s)
        return NotImplemented

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, v: Vec3) -> float:
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v: Vec3) -> Vec3:
        return Vec3(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def potto_norm(self):
        return Sqrt(expr=self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> Vec3:
        return self / self.norm()

    def l1(self) -> float:
        return abs(self.x) + abs(self.y) + abs(self.z)

    def linf(self) -> float:
        return max(abs(self.x), abs(self.y), abs(self.z))
