from __future__ import annotations
from typing import Any, TypeVar
import builtins
from dataclasses import dataclass
from abc import abstractmethod, ABC
import math


@dataclass(frozen=True)
class AbstractVector2(ABC):
    x: float = 0.
    y: float = 0.

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __iter__(self):
        return iter((self.x, self.y))


@dataclass(frozen=True)
class Pnt2(AbstractVector2):
    """
    Points of a (2d) Euclidean affine plane
    Immutable

    Pnt2(0, 0) is the origin
    """
    x: float = 0.
    y: float = 0.

    def __add__(self, v: Any) -> Pnt2:
        match v:
            case Pnt2():
                return NotImplemented  # NOTE: We disallow adding Points to Points, that's the main difference with Vec
            case Vec2(x, y):
                return Pnt2(self.x + x, self.y + y)
        return NotImplemented

    def __sub__(self, v: Any) -> Pnt2 | Vec2:
        match v:
            case Pnt2(x, y):
                return Vec2(self.x - x, self.y - y)  # Pnt - Pnt => Vec
            case Vec2(x, y):
                return Pnt2(self.x - x, self.y - y)  # Pnt - Vec => Pnt
        return NotImplemented

    def __mul__(self, s: Any) -> Pnt2:
        match s:
            case builtins.int | builtins.float:
                return Pnt2(self.x * s, self.y * s)  # identifying [point * s] with [origin + (point - origin)*s]
        return NotImplemented

    def __rmul__(self, s: Any) -> Pnt2:
        return self * s

    def __truediv__(self, s: Any) -> Pnt2:
        match s:
            case builtins.int | builtins.float:
                return Pnt2(self.x / s, self.y / s)
        return NotImplemented

    def __neg__(self) -> Pnt2:
        return Pnt2(-self.x, -self.y)


@dataclass(frozen=True)
class Vec2(AbstractVector2):
    """
    Vectors from a (2d) real Euclidean vector space equipped with inner product and induced metric
    Immutable

    Secretly we can also use math.inf for extended reals
    """
    x: float = 0.
    y: float = 0.

    def __add__(self, v: Any) -> Vec2:
        match v:
            case Pnt2(x, y) | Vec2(x, y):
                return Vec2(self.x - x, self.y - y)
        return NotImplemented

    def __sub__(self, v: Any) -> Vec2:
        match v:
            case Vec2(x, y):
                return Vec2(self.x - x, self.y - y)
        return NotImplemented

    def __mul__(self, s: Any) -> Vec2:
        match s:
            case builtins.int | builtins.float:
                return Vec2(self.x * s, self.y * s)
        return NotImplemented

    def __rmul__(self, s: Any) -> Vec2:
        return self * s

    def __truediv__(self, s: Any) -> Vec2:
        match s:
            case builtins.int | builtins.float:
                return Vec2(self.x / s, self.y / s)
        return NotImplemented

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def dot(self, v: Vec2) -> float:
        return self.x * v.x + self.y * v.y

    def cross(self) -> Vec2:
        return Vec2(-self.y, self.x)

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> Vec2:
        return self / self.norm()

    def l1(self) -> float:
        return abs(self.x) + abs(self.y)

    def linf(self) -> float:
        return max(abs(self.x), abs(self.y))
