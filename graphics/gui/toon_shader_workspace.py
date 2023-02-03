import dearpygui.dearpygui as dpg
import math
from dataclasses import dataclass
from graphics.scene.scenegraph import Viewport, SceneGraph, SceneNode
from graphics.scene.raytrace import RayTrace, RayTraceDirectPotto, RayTraceDirectPython, RayTracePotto, RayTracePython, RayTraceSwappablePotto
from graphics.scene.camera import TurntableCamera
from graphics.gui.image import PixGrid
from graphics.util.diffeo_util import make_diffeo
import numpy as np

from graphics.util.color import Colors
from graphics.util.mat3 import Tform3
from graphics.util.ray3 import Ray3
from graphics.util.vec3 import Pnt3, Vec3
from graphics.scene.shape3 import Triangle, Sphere, VAR_T, VAR_VAL
from graphics.scene.light import PointLight

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
                c[xpix, ypix] = [dv * num_xpix, v * num_xpix * num_ypix, 0, 1]
                print(f"\t({xpix}, {ypix}): {c[xpix, ypix]}")
        raypix.img = c

    # unittest()
    # rasterize()
    raytrace()


def main():
    # np.random.seed(1)
    dpg.create_context()
    dpg.create_viewport(title='Toon Shader Demo', width=1800, height=800)
    dpg.setup_dearpygui()

    def to_raw_pnt(var_vals, p: Pnt3):
        def ev(e):
            return evaluate_all(e, var_vals)
        return Pnt3(ev(p.x), ev(p.y), ev(p.z))

    sg = SceneGraph(
        [
            SceneNode(
                transform=Tform3(Vec3(0, 0, 0)),
                shapes=[
                    Triangle(
                        Pnt3(Const(-8), Const(-8), Const(0 + 1e-3)),
                        Pnt3(Const(8) * VAR_T, Const(-8), Const(0 - 1e-4)),
                        Pnt3(Const(-8), Const(8) * VAR_T, Const(-1e-3)),
                        e=Colors.WHITE,
                    ),
                ],
            ),
            # SceneNode(
            #     transform=Tform3(Vec3(0, 0, 0)),
            #     shapes=[
            #         Sphere(
            #             rad=8, p=Pnt3(0, 0, 0), e=Colors.WHITE
            #         ),
            #     ]
            # ),
        ],
        [
            PointLight(
                to_raw_pnt(VAR_VAL, Pnt3(Const(0), Const(0), Const(5.0))),
                intensity=Colors.WHITE,
            )
        ]
    )
    raw_sg = SceneGraph(
        [
            SceneNode(
                transform=Tform3(Vec3(0, 0, 0)),
                shapes=[
                    Triangle(
                        to_raw_pnt(VAR_VAL, Pnt3(Const(-8), Const(-8), Const(0 + 1e-3))),
                        to_raw_pnt(VAR_VAL, Pnt3(Const(8) * VAR_T, Const(-8), Const(0 - 1e-4))),
                        to_raw_pnt(VAR_VAL, Pnt3(Const(-8), Const(8) * VAR_T, Const(-1e-3))),
                        e=Colors.WHITE,
                    ),
                ],
            ),
            # SceneNode(
            #     transform=Tform3(Vec3(0, 0, 0)),
            #     shapes=[
            #         Sphere(
            #             rad=8, p=Pnt3(0, 0, 0), e=Colors.WHITE
            #         ),
            #     ]
            # ),
        ],
        [
            PointLight(
                to_raw_pnt(VAR_VAL, Pnt3(Const(0), Const(0), Const(5.0))),
                intensity=Colors.WHITE,
            )
        ]
    )
    num_xpix = 8
    num_ypix = 8
    aspect_ratio = num_xpix / num_ypix
    cam = TurntableCamera(dist=25, angle=0, aspect_ratio=aspect_ratio, fov_angle_y=90*np.pi/180)
    vp = Viewport(sg, cam)
    rt = RayTrace(sg, cam, 10)
    rt2 = RayTracePython(raw_sg, cam, 10)
    rt3 = RayTracePotto(sg, cam, 10)
    rt4 = RayTraceDirectPython(raw_sg, cam, 10)
    rt5 = RayTraceDirectPotto(sg, cam, 10)
    rt6 = RayTraceSwappablePotto(sg, cam, 10)
    raypix = PixGrid(width=num_xpix, height=num_ypix)
    ref_raypix = PixGrid(width=num_xpix, height=num_ypix)
    is_default_shader = False

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
                    proj_mat = dpg.create_perspective_matrix(fov=cam.fov_angle_y, aspect=cam.aspect_ratio, zNear=0.1, zFar=100)
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
        nonlocal is_default_shader
        is_default_shader = not is_default_shader
        print(f"do_render was called, is_default_shader = {is_default_shader}")
        # raypix.img = rt3.ray_trace_toon_parallel(raypix.width, raypix.height)  # Sequential code
        raypix.img = rt6.ray_trace_toon_parallel(raypix.width, raypix.height, is_default_shader)
        # raypix.img = rt5.ray_trace_toon_parallel(raypix.width, raypix.height)  # Parallel code
        raypix.update_raw_array()
        print("do_render has finished")

    def ref_do_render():
        print("do_render was called (reference)")
        # TODO: ortho camera for simple example first?
        # hack(raypix, raypix.width, raypix.height, cam)  # Bad reference
        # ref_raypix.img = rt4.ray_trace_toon_hack(raypix.width, raypix.height)  # Actual code
        # ref_raypix.img = rt3.ray_trace_toon_parallel(raypix.width, raypix.height)
        ref_raypix.img = rt2.ray_trace_toon_hack(ref_raypix.width, ref_raypix.height)  # Eventual code for fast demo
        ref_raypix.update_raw_array()
        print("do_render has finished (reference)")

    do_render()
    ref_do_render()
    with dpg.window(label="raytracer", width=550, height=550) as raytrace_id:
        dpg.set_item_pos(raytrace_id, [600, 0])
        dpg.add_image(raypix.gui_id, width=500, height=500)  # TODO fix the default antialiasing

    with dpg.window(label="raytrace params") as do_render_id:
        button_id = dpg.add_button(label="do render", callback=do_render)
        dpg.set_item_pos(do_render_id, [630, 580])

    with dpg.window(label="python ref", width=550, height=550) as python_ref_id:
        dpg.set_item_pos(python_ref_id, [1200, 0])
        dpg.add_image(ref_raypix.gui_id, width=500, height=500)  # TODO fix the default antialiasing

    with dpg.window(label="raytrace params") as ref_do_render_id:
        button_id = dpg.add_button(label="do render", callback=ref_do_render)
        dpg.set_item_pos(ref_do_render_id, [1230, 580])

    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
