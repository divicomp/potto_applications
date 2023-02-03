from __future__ import annotations
from dataclasses import dataclass
from abc import abstractmethod, ABC
from enum import Enum
import numpy as np
from numpy.random import rand

from graphics.util.color import Color, Colors
from graphics.util.ray3 import Ray3, Hit3
from graphics.util.vec3 import Pnt3, Vec3

from potto import evaluate_all, VarVal, Var, TegVar, GExpr


VAR_S = Var("s")
VAR_T = Var("t")
VAR_VAL = VarVal({VAR_T.name: 1, VAR_S.name: -1})


class RenderMode(Enum):
    BASIC = 0
    SELECTED = 1


@dataclass
class Shape3(ABC):  # TODO create subclasses
    """
    A 3d shape
    """

    @abstractmethod
    def render(self, dpg, mode=RenderMode.BASIC):
        raise NotImplementedError

    @abstractmethod
    def intersect(self, r: Ray3) -> Hit3:
        raise NotImplementedError

    @abstractmethod
    def get_normal(self, hit: Hit3) -> Vec3:
        raise NotImplementedError

    @abstractmethod
    def get_raw_normal(self, hit: Hit3) -> Vec3:
        raise NotImplementedError


@dataclass
class Quad(Shape3):
    """
    An 3d quadrilateral shape, with points specified in draw order
    """

    p0: Pnt3
    p1: Pnt3
    p2: Pnt3
    p3: Pnt3
    e: Color = Colors.TRANSPARENT
    c: Color = Colors.WHITE

    def render(self, dpg, mode=RenderMode.BASIC):
        dpg.draw_quad(
            self.p0.to_tuple(),
            self.p1.to_tuple(),
            self.p2.to_tuple(),
            self.p3.to_tuple(),
            color=(0, 0, 0),
            fill=(0, 255, 255, 150),
        )

    def intersect(self, r: Ray3) -> Hit3:
        t1 = Triangle(self.p0, self.p1, self.p2)
        t2 = Triangle(self.p0, self.p2, self.p3)
        hit = t1.intersect(r)
        if not hit.t:
            return t2.intersect(r)
        return hit

    def get_normal(self, hit: Hit3) -> Vec3:
        assert hit.t is not None, "Hit has to happen for there to be a normal"
        t1 = Triangle(self.p0, self.p1, self.p2)
        t2 = Triangle(self.p0, self.p2, self.p3)
        check_hit = t1.intersect(hit.r)
        if not check_hit.t:
            return t2.get_normal(hit)
        return t1.get_normal(hit)

    def get_raw_normal(self, hit: Hit3) -> Vec3:
        assert hit.t is not None, "Hit has to happen for there to be a normal"
        t1 = Triangle(self.p0, self.p1, self.p2)
        t2 = Triangle(self.p0, self.p2, self.p3)
        check_hit = t1.intersect(hit.r)
        if not check_hit.t:
            return t2.get_raw_normal(hit)
        return t1.get_raw_normal(hit)


@dataclass
class Triangle(Shape3):
    """
    An 3d triangle shape
    Vertices are specified in clockwise orientation around the plane normal
    """

    p0: Pnt3
    p1: Pnt3
    p2: Pnt3
    e: Color = Colors.WHITE
    c: Color = Colors.WHITE

    def render(self, dpg, mode=RenderMode.BASIC):
        def eval_pnt(p, var_val):
            match p.x:
                case GExpr():
                    px = evaluate_all(p.x, VAR_VAL)
                    py = evaluate_all(p.y, VAR_VAL)
                    pz = evaluate_all(p.z, VAR_VAL)
                    return Pnt3(px, py, pz)
            return p

        p0 = eval_pnt(self.p0, VAR_VAL)
        p1 = eval_pnt(self.p1, VAR_VAL)
        p2 = eval_pnt(self.p2, VAR_VAL)
        dpg.draw_triangle(
            p0.to_tuple(),
            p1.to_tuple(),
            p2.to_tuple(),
            color=(0, 0, 0),
            fill=(0, 255, 255, 150),
        )

    def intersect(self, r: Ray3) -> Hit3:
        # Adaptation of
        # https://www.erikrotteveel.com/python/three-dimensional-ray-tracing-in-python/
        u = self.p1 - self.p0
        v = self.p2 - self.p0

        normal = self.get_normal(Hit3())

        denom = normal.dot(r.d)
        num = normal.dot(self.p0 - r.o)

        if denom == 0:
            # The ray is precisely perpendicular to normal along the plane
            if num != 0:
                # The ray is parallel to the plane and outside it
                return Hit3(r, None)
            t = 0
        else:
            # The ray hits the plane
            t = num / denom

        # The hit never happens because time >= 0
        if t < 0:
            return Hit3(r, None)

        # The ray to the plane is Ray3(r.o, k * r.d)
        # Is that point in the triangle?
        w = r.o + t * r.d - self.p0

        denom = u.dot(v) ** 2 - u.dot(u) * v.dot(v)

        si = (u.dot(v) * w.dot(v) - v.dot(v) * w.dot(u)) / denom
        if (si < 0.0) | (si > 1.0):
            return Hit3()

        ti = (u.dot(v) * w.dot(u) - u.dot(u) * w.dot(v)) / denom
        if (ti < 0.0) | (si + ti > 1.0):
            return Hit3()

        return Hit3(r, t)

    def get_normal(self, _: Hit3) -> Vec3:
        u = self.p1 - self.p0
        v = self.p2 - self.p0
        w = u.cross(v)
        return w / w.norm()


    def get_raw_normal(self, hit: Hit3) -> Vec3:
        u = self.p1 - self.p0
        v = self.p2 - self.p0
        w = u.cross(v)
        return w

@dataclass
class Sphere(Shape3):
    rad: float
    p: Pnt3
    e: Color = Colors.WHITE
    c: Color = Colors.WHITE

    def render(self, dpg, mode=RenderMode.BASIC):
        # approximate sphere with polys

        dpg.draw_circle(
            self.p.to_tuple(),
            radius=self.rad,
            color=(0, 0, 0),
            fill=(0, 255, 255, 150),
        )
        num_lat = 8
        num_long = 12
        thetas = np.linspace(0, np.pi, num=num_lat)
        phis = np.linspace(0, 2*np.pi, num=num_long)
        ps = np.zeros((num_lat, num_long, 3))
        r = self.rad
        for i, theta in enumerate(thetas):
            for j, phi in enumerate(phis):
                x = r * np.cos(phi) * np.sin(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(theta)
                ps[i, j] = [x, y, z]

        for p0s, p1s, p2s, p3s in zip(ps, ps[1:, :], ps[:, 1:], ps[1:, 1:]):
            for p0, p1, p2, p3 in zip(p0s, p1s, p2s, p3s):
                    dpg.draw_triangle(
                        p0, p1, p2,
                        color=(0, 0, 0),
                        fill=(0, 255, 255, 150),
                    )
                    dpg.draw_triangle(
                        p1, p2, p3,
                        color=(0, 0, 0),
                        fill=(0, 255, 255, 150),
                    )
        # dpg.draw_circle(
        #     self.p,
        #     radius=self.rad,
        #     color=(0, 0, 0),
        #     fill=(0, 255, 255, 150),
        # )

    def intersect(self, r: Ray3) -> Hit3:
        p, rad = self.p, self.rad
        op: Vec3 = p - r.o
        eps = 1e-4
        b = op.dot(r.d)
        determinant = b * b - op.dot(op) + rad * rad
        if determinant < 0:
            return Hit3(r, None)
        sqrt_det = np.sqrt(determinant)

        t = b - sqrt_det
        if t > eps:
            return Hit3(r, t)
        t = b + sqrt_det
        return Hit3(r, t) if t > eps else Hit3(r, None)

    def get_normal(self, hit: Hit3) -> Vec3:
        assert hit.t is not None, "Hit has to happen for there to be a normal"
        x = hit.r.o + hit.r.d * hit.t
        to_p: Vec3 = x - self.p
        return to_p / to_p.norm()

    def get_raw_normal(self, hit: Hit3) -> Vec3:
        assert hit.t is not None, "Hit has to happen for there to be a normal"
        x = hit.r.o + hit.r.d * hit.t
        to_p: Vec3 = x - self.p
        return to_p
