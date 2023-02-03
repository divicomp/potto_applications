from __future__ import annotations
from typing import Any, TypeVar
import builtins
from dataclasses import dataclass
from abc import abstractmethod, ABC
import math

from graphics.util.vec3 import Pnt3, Vec3
from graphics.util.mat3 import Tform3


@dataclass(frozen=True)
class Ray3(ABC):
    """
    A ray from a point going in a certain direction
    """
    o: Pnt3 = Pnt3()
    d: Vec3 = Vec3(0, 0, 1)

    def __str__(self):
        return f'Ray{{{self.o} + t{self.d}}}'

    def __rmatmul__(self, t: Any) -> Ray3:
        match t:
            case Tform3():
                return Ray3(t @ self.o, t @ self.d)
        return NotImplemented


@dataclass(frozen=True)
class Hit3(ABC):
    """
    A particular point on a ray
    """
    r: Ray3 = Ray3()
    t: float | None = None

    def __str__(self):
        match self.t:
            case float(t):
                return f'Hit{{t={t} at {self.hitpoint()}}}'
            case None:
                return f'[No hit]'

    def hitpoint(self):
        return self.r.o + self.t * self.r.d
