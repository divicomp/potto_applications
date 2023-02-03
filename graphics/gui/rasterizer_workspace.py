import dearpygui.dearpygui as dpg
import math
from dataclasses import dataclass
from graphics.scene.scenegraph import Viewport, SceneGraph, SceneNode
from graphics.scene.raytrace import RayTrace
from graphics.scene.camera import TurntableCamera
from graphics.gui.image import PixGrid
from graphics.util.diffeo_util import make_diffeo
import numpy as np

from graphics.util.color import Colors
from graphics.util.mat3 import Tform3
from graphics.util.ray3 import Ray3
from graphics.util.vec3 import Pnt3, Vec3
from graphics.scene.shape3 import Triangle, VAR_T

from potto import (
    Int,
    Const,
    Var,
    GExpr,
    TegVar,
    Heaviside,
    Diffeomorphism,
    VarVal,
)
from potto import deriv, evaluate_all, simplify
from potto.test import Uniform


def hack(raypix, num_xpix, num_ypix, perspective_cam):
    class Shift(Diffeomorphism):
        """
        f(x, y) = t - x - y, y
        f-1(z, y) = t - z - y, y
        """

        def __str__(self):
            return f"[{self.tvars[0].name} - {self.vars[0].name} - {self.vars[1].name}]"

        def function(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
            t = vars[0]
            x, y = tvars[0], tvars[1]
            return (t - x - y, y)

        def inverse(self, vars: tuple[GExpr, ...], tvars: tuple[GExpr, ...]) -> tuple[GExpr, ...]:
            t = vars[0]
            z, y = tvars[0], tvars[1]
            return (t - z - y, y)

    def unittest():
        t = Var("t")
        dt = Var("dt")
        x = TegVar("x")
        y = TegVar("y")
        mu = Uniform(0, 1, x)
        nu = Uniform(0, 1, y)
        integrand = Heaviside(Shift((t,), (x, y), Const(1)))
        # int_0^1 dy  int_0^1 dx  H(t - x - y)
        e = Int(Int(integrand, mu), nu)
        e = simplify(e)
        var_val = VarVal({t.name: 1})
        v = evaluate_all(e, var_val, num_samples=1000)

        dexpr = deriv(e, {t.name: dt.name})
        dexpr = simplify(dexpr)
        dvar_val = VarVal({t.name: 1, dt.name: 1})
        # d/dt int_0^1 dy  int_0^1 dx  H(t - x - y)
        # int_0^1 dy  int_0^1 dx  Delta(t - x - y) dt
        # int_0^1 dy  int_0^1 dx  [x = t-y] dt
        # int_0^1 dy  [0 <= t-y <= 1] dt
        # int_0^1 dy  [0 <= 1-y <= 1] dt
        # int_0^1 dy  [0 <= y <= 1] dt
        # 1
        dv = evaluate_all(dexpr, dvar_val, num_samples=100)

        print(f"\tv: {v}\tdv: {dv}")

    def rasterize():
        screen_width = 1
        screen_height = 1

        t = Var("t")
        dt = Var("dt")
        x = TegVar("x")
        y = TegVar("y")
        integrand = Heaviside(Shift((t,), (x, y), Const(1)))

        def rasterize_xy(xpix, ypix) -> tuple[float, float]:
            xlo = xpix / num_xpix * screen_width
            xhi = (xpix + 1) / num_xpix * screen_width
            ylo = ypix / num_ypix * screen_height
            yhi = (ypix + 1) / num_ypix * screen_height

            mu = Uniform(xlo, xhi, x)
            nu = Uniform(ylo, yhi, y)
            # int dy  int dx  H(t - x - y)
            e = Int(Int(integrand, mu), nu)
            # e = simplify(e)
            var_val = VarVal({t.name: 1})
            v = evaluate_all(e, var_val, num_samples=1000)

            dexpr = deriv(e, {t.name: dt.name})
            # dexpr = simplify(dexpr)
            dvar_val = VarVal({t.name: 1, dt.name: 1})
            dv = evaluate_all(dexpr, dvar_val, num_samples=100)
            return v, dv

        c = np.zeros((num_xpix, num_ypix, 4))
        for xpix in range(num_xpix):
            for ypix in range(num_ypix):
                v, dv = rasterize_xy(xpix, ypix)
                c[xpix, ypix] = [0, v * num_xpix, dv * num_xpix * num_ypix, 1]
                print(f"\t({xpix}, {ypix}): {c[xpix, ypix]}")
        raypix.img = c

    def raytrace():
        screen_width = 1
        screen_height = 1

        t = Var("t")
        dt = Var("dt")
        x = TegVar("x")
        y = TegVar("y")

        def rasterize_xy(xpix, ypix) -> tuple[float, float]:
            xlo = xpix / num_xpix * screen_width
            xhi = (xpix + 1) / num_xpix * screen_width
            ylo = ypix / num_ypix * screen_height
            yhi = (ypix + 1) / num_ypix * screen_height

            # TODO (kmu): fix transforms, it's slightly off
            # TODO (kmu): fix backfaces
            ray_o = perspective_cam.transform @ Pnt3(x, y, Const(0))
            ray_d = perspective_cam.transform @ Vec3(Const(0), Const(0), Const(-1))
            ray = Ray3(ray_o, ray_d)
            # ray_t = Const(1)
            plane_n = Vec3(Const(0), Const(0), Const(1))
            plane_q = Vec3(t, Const(0), Const(0))
            plane_m = Vec3(Const(-1), Const(-1), Const(0))

            """
            Ray
            o = (x, y, 1)
            d = (0, 0, -1)
            => t = 1
            
            Plane
            n = (0, 0, 1)
            q = (t, 0, 0)
            m = (-1, -1, 0)
            
            
            ((o + t * d) - q) dot n == 0
            ((o + t * d) - q) dot m > 0
            """

            @dataclass(frozen=True)
            class HackDot(object):
                px: GExpr
                py: GExpr
                pz: GExpr

                # TODO: rmul priorities
                def __mul__(self, other):
                    match other:
                        case HackDot(qx, qy, qz):
                            return self.px * qx + self.py * qy + self.pz * qz
                        case GExpr():
                            return HackDot(self.px * other, self.py * other, self.pz * other)
                        case _:
                            raise ValueError(f"Cannot multiply Hackdot with {type(other)} object: {other}")

                def __add__(self, other):
                    match other:
                        case HackDot(qx, qy, qz):
                            return HackDot(self.px + qx, self.py + qy, self.pz + qz)
                        case _:
                            raise ValueError(f"Cannot add Hackdot with {type(other)} object: {other}")

                def __sub__(self, other):
                    match other:
                        case HackDot(qx, qy, qz):
                            return HackDot(self.px - qx, self.py - qy, self.pz - qz)
                        case _:
                            raise ValueError(f"Cannot subtract Hackdot with {type(other)} object: {other}")

            qh = HackDot(plane_q.x, plane_q.y, plane_q.z)
            oh = HackDot(ray.o.x, ray.o.y, ray.o.z)
            nh = HackDot(plane_n.x, plane_n.y, plane_n.z)
            mh = HackDot(plane_m.x, plane_m.y, plane_m.z)
            dh = HackDot(ray.d.x, ray.d.y, ray.d.z)
            ray_t = ((qh - oh) * nh) / (dh * nh)
            diff_expr = (oh - qh + dh * ray_t) * mh
            diff_expr = simplify(diff_expr)

            # integrand = Heaviside(Shift([t], [x, y], Const(1)))
            # integrand = Heaviside(make_diffeo(t - x - y, [t], [x, y]))
            integrand = Heaviside(make_diffeo(diff_expr, (t,), (x, y)))
            mu = Uniform(xlo, xhi, x)
            nu = Uniform(ylo, yhi, y)
            # int dy  int dx  H(t - x - y)
            e = Int(Int(integrand, mu), nu)
            e = simplify(e)
            var_val = VarVal({t.name: 1})
            v = evaluate_all(e, var_val, num_samples=100)

            dexpr = deriv(e, {t.name: dt.name})
            dexpr = simplify(dexpr)
            dvar_val = VarVal({t.name: 1, dt.name: 1})
            dv = evaluate_all(dexpr, dvar_val, num_samples=100)
            return v, dv

        c = np.zeros((num_xpix, num_ypix, 4))
        for xpix in range(num_xpix):
            for ypix in range(num_ypix):
                v, dv = rasterize_xy(xpix, ypix)
                c[xpix, ypix] = [dv * num_xpix * num_ypix, v * num_xpix * num_ypix, 0, 1]
                print(f"\t({xpix}, {ypix}): {c[xpix, ypix]}")
        raypix.img = c

    # unittest()
    # rasterize()
    raytrace()


def main():
    # np.random.seed(1)
    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()

    sg = SceneGraph(
        [
            SceneNode(
                # transform=Tform3(Vec3(2, 1, 0)),
                transform=Tform3(Vec3(0, 0, 0)),
                shapes=[
                    Triangle(
                        Pnt3(Const(-1), Const(-1), Const(0)),
                        Pnt3(Const(1) * VAR_T, Const(-1), Const(0)),
                        Pnt3(Const(-1), Const(1) * VAR_T, Const(-0)),
                        e=Colors.WHITE,
                    ),
                ],
            )
        ]
    )
    cam = TurntableCamera(dist=1, angle=0)
    vp = Viewport(sg, cam)
    rt = RayTrace(sg, cam, 10)
    raypix = PixGrid(width=10, height=10)

    def set_turntable_angle(sender, turntable_angle):
        vp.camera.angle = turntable_angle
        view = dpg.create_lookat_matrix(
            cam.transform.translate.to_tuple(),
            Pnt3(0, 0, 0).to_tuple(),
            Vec3(0, 1, 0).to_tuple(),
        )  # TODO abstract into camera class
        dpg.apply_transform("turntable camera", view)

    with dpg.window(label="viewport", width=550, height=550):
        with dpg.drawlist(width=500, height=500):
            with dpg.draw_layer(
                tag="CLIP",
                depth_clipping=False,
                perspective_divide=True,
                cull_mode=dpg.mvCullMode_None,
            ) as clip_id:
                dpg.set_clip_space(clip_id, 0, 0, 500, 500, -1.0, 1.0)

                with dpg.draw_node(tag="camera proj") as proj_id:
                    fov = 90  # TODO: abstract into camera code
                    proj_mat = dpg.create_perspective_matrix(fov=fov * math.pi / 180.0, aspect=1.0, zNear=0.1, zFar=100)
                    dpg.apply_transform(proj_id, proj_mat)

                    with dpg.draw_node(tag="turntable camera"):
                        set_turntable_angle(None, 0)

                        # axes arrows
                        dpg.draw_arrow((4, 0, 0), (0, 0, 0), color=(255, 0, 0, 150), size=1)
                        dpg.draw_arrow((0, 4, 0), (0, 0, 0), color=(0, 255, 0, 150), size=1)
                        dpg.draw_arrow((0, 0, 4), (0, 0, 0), color=(0, 0, 255, 150), size=1)

                        # draw scenegraph
                        vp.scene.render(dpg)

    with dpg.window(label="hack camera") as camera_param_id:
        slider_id = dpg.add_slider_float(
            label="turn angle",
            callback=set_turntable_angle,
            min_value=-180,
            max_value=180,
            width=100,
        )
        dpg.set_item_pos(camera_param_id, [30, 580])

    def do_render():
        print("do_render was called")
        hack(raypix, raypix.width, raypix.height, cam)
        # raypix.img = rt.diff_ray_trace(raypix.width, raypix.height)
        # raypix.img = rt.diff_ray_trace_parallel(raypix.width, raypix.height)
        raypix.update_raw_array()
        print("do_render has finished")

    with dpg.window(label="raytracer", width=550, height=550) as raytrace_id:
        dpg.set_item_pos(raytrace_id, [600, 0])
        dpg.add_image(raypix.gui_id, width=500, height=500)  # TODO fix the default antialiasing

    with dpg.window(label="raytrace params") as camera_param_id:
        button_id = dpg.add_button(label="do render", callback=do_render)
        dpg.set_item_pos(camera_param_id, [630, 580])

    # set_turntable_angle(None, 45)
    do_render()
    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
