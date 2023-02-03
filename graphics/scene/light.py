from __future__ import annotations
from dataclasses import dataclass
from abc import abstractmethod, ABC
from graphics.util.vec3 import Pnt3, Vec3
from graphics.util.color import Color
import math


@dataclass
class Light(ABC):

    @abstractmethod
    def get_light_dir(self, p: Pnt3) -> Vec3:
        raise NotImplementedError

    @abstractmethod
    def get_color(self) -> Color:
        raise NotImplementedError

    @abstractmethod
    def get_raw_light_dir(self, p: Pnt3) -> Vec3:
        raise NotImplementedError


@dataclass
class PointLight(ABC):
    pos: Pnt3
    intensity: Color

    def get_light_dir(self, p: Pnt3) -> Vec3:
        w = self.pos - p
        norm = w.dot(w)
        norm = math.sqrt(norm) 
        return w / norm

    def get_color(self) -> Color:
        return self.intensity

    def get_raw_light_dir(self, p: Pnt3) -> Vec3:
        return self.pos - p
