from __future__ import annotations
from dataclasses import dataclass, field
from abc import abstractmethod, ABC
from enum import Enum
import math
import array

import dearpygui.dearpygui as dpg
import numpy as np

from graphics.util.ray3 import Ray3, Hit3
from graphics.util.mat3 import Tform3, Mat3
from graphics.util.vec3 import Pnt3, Vec3


@dataclass()
class PixGrid(ABC):
    width: int = 1024
    height: int = 768
    img: np.ndarray = np.zeros(0, dtype=float)

    def __post_init__(self):
        if self.img.shape != (self.width, self.height, 4):
            self.img = np.full((self.width, self.height, 4), [1, 0, 1, 1])
        with dpg.texture_registry():
            self.raw_img = array.array('f', self.img.flatten().tolist())
            self.update_raw_array()
            self.gui_id = dpg.add_raw_texture(self.width, self.height, self.raw_img, format=dpg.mvFormat_Float_rgba)

    def update_raw_array(self):  # TODO get numpy arrays to directly work
        new_raw = np.flip(self.img, axis=1).swapaxes(0, 1).flatten().tolist()
        for i in range(len(self.raw_img)):
            self.raw_img[i] = new_raw[i]
