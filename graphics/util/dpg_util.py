from graphics.util.mat3 import Tform3
from dearpygui._dearpygui import mvMat4


def to_mvmat4(transform: Tform3):
    t = transform.translate
    rs = transform.rot_scale
    return mvMat4(rs[0, 0], rs[0, 1], rs[0, 2], t[0], rs[1, 0], rs[1, 1], rs[1, 2], t[1], rs[2, 0], rs[2, 1], rs[2, 2], t[2], 0., 0., 0., 1.)
