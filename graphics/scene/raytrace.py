from __future__ import annotations
from typing import Any, List
from dataclasses import dataclass, field
from collections.abc import Iterable
from itertools import chain
from multiprocessing import cpu_count
from joblib import Parallel, delayed

from tqdm import trange
import numpy as np
from numpy.random import rand

from graphics.util.diffeo_util import make_diffeo
from graphics.scene.camera import Camera
from graphics.scene.scenegraph import SceneGraph
from graphics.util.ray3 import Ray3, Hit3
from graphics.util.mat3 import Tform3, Mat3
from graphics.util.vec3 import Pnt3, Vec3
from graphics.util.vec2 import Pnt2
from graphics.util.color import Colors, Color
from graphics.scene.shape3 import VAR_S, VAR_T, Triangle
from graphics.scene.light import PointLight

from potto import (
    Measure,
    Sym,
    Var,
    GExpr,
    TegVar,
    Heaviside,
    Diffeomorphism,
    Mul,
    Div,
    Abs,
    Sample,
    VarVal,
    Const,
    BoundedLebesgue,
    Int,
    simplify,
    deriv,
    evaluate_all,
    Sqrt,
    Function,
    App
)

deltax = TegVar("deltax")
deltay = TegVar("deltay")
pix_x = Var("pix_x")
pix_y = Var("pix_y")
M_PI = 3.1415926535

@dataclass
class RayTrace(object):
    """
    A 3d ray tracer that traces rays to render a scene
    """

    scene: SceneGraph
    camera: Camera
    num_samples: int = 10
    cur_screen_width = 0
    cur_screen_height = 0

    def build_diff_rt_expr(self, pix_x, pix_y, screen_width, screen_height) -> tuple[GExpr, GExpr, GExpr]:
        raise NotImplementedError


@dataclass
class RayTracePython(RayTrace):
    """
    Reference renderer
    """

    def radiance(self, r):
        # test if hit anything in scenegraph
        hit_nodes = list(self.scene.pick(r))  # no acceleration structure for simplicity for now
        if not hit_nodes:
            return Colors.BLACK
        closest_hit_node, node_hit = min(hit_nodes, key=lambda nh: nh[1])  # TODO return shape that was hit
        hit_shape = closest_hit_node.shapes[0]

        # material emmission
            # lambert
                # normal dot light ray
                # need lights
                    # for now hardcode a light somewhere just to get stuff working?
            # edge shader
                # normal dot look angle < threthhold  # later make this based on screen space thickness?
                # make sphere
        sample_color = Colors.RED
        threshold = 0.3
        if abs(hit_shape.get_normal(node_hit).dot(r.d)) < threshold:
            sample_color = Colors.BLUE  # only on first bounce?
        # do brdf
        # send out more rays
        # TODO implement


        return sample_color

    def ray_trace_xy(self, pix_x, pix_y, max_xpix, max_ypix) -> Any:
        """
        screen coords: [0, max_xpix] x [0, max_ypix]
        samples in: [pix_x, pix_x+1] x [pix_y, pix_y+1]
        ndc: [-1, 1] x [-1, 1]
        camera coords: (local coords of camera sensor)
        """

        pix_color = Colors.TRANSPARENT
        x_samples = np.random.uniform(pix_x, pix_x+1, size=self.num_samples)
        y_samples = np.random.uniform(pix_y, pix_y+1, size=self.num_samples)
        # integrate over screen pix -> send rays out in screen coords
        for pix_xsamp, pix_ysamp in zip(x_samples, y_samples):
            # convert screen coords to camera
            # convert camera coords to world
            look_ray = self.camera.ndc_to_ray(
                self.camera.screen_point_to_ndc(Pnt2(pix_xsamp, pix_ysamp), max_xpix, max_ypix))
            # acc color
            ray_radiance_color = self.radiance(look_ray)
            pix_color += ray_radiance_color

        # ret color
        return pix_color.tolist()

    def ray_trace(self, screen_width, screen_height):
        c = np.zeros((screen_width, screen_height, 4))  # RGBA
        for x in range(screen_width):
            for y in range(screen_height):
                c[x, y] = self.ray_trace_xy(pix_x=x, pix_y=y, max_xpix=screen_width, max_ypix=screen_height)
                # print(c[x, y])
                print(f"traced {x, y} {c[x, y]}")
        return c

    def ray_trace_parallel(self, screen_width, screen_height):
        def ray_trace_xy_proxy(x, y):
            c = self.ray_trace_xy(pix_x=x, pix_y=y, max_xpix=screen_width, max_ypix=screen_height)
            print((x, y), c)
            return (x, y), c

        results = Parallel(n_jobs=cpu_count() - 1)(
            delayed(ray_trace_xy_proxy)(x, y) for x in range(screen_width) for y in range(screen_height)
        )

        c = np.zeros((screen_width, screen_height, 4))
        for k, v in results:
            c[k] = v
        return c


@dataclass
class RayTracePotto(RayTrace):
    """
    Potto Shader Implementation
    """
    def radiance(self, r: Ray3, pix_x: Var, pix_y: Var, depth: int = 0) -> Color:
        ray = r
        vars = (pix_x, pix_y)
        tvars = (deltax, deltay)
        all_hits = []
        for n in self.scene.nodes:
            for shape in n.shapes:
                tri: Triangle = shape
                nt = n.transform.inverse()
                v = Vec3(r.o.x, r.o.y, r.o.z)
                matmulofro = Pnt3(nt.rot_scale.x0 * v, nt.rot_scale.x1 * v, nt.rot_scale.x2 * v)
                newro = matmulofro + nt.translate

                v = Vec3(r.d.x, r.d.y, r.d.z)
                matmulofrd = Pnt3(nt.rot_scale.x0 * v, nt.rot_scale.x1 * v, nt.rot_scale.x2 * v)
                r = Ray3(newro, matmulofrd)

                u = tri.p1 - tri.p0
                v = tri.p2 - tri.p0

                wx = u.y * v.z - u.z * v.y
                wy = u.z * v.x - u.x * v.z
                wz = u.x * v.y - u.y * v.x

                denom = r.d.x * wx + r.d.y * wy + r.d.z * wz  # Linear in TegVars
                denom = simplify(denom)

                qx = tri.p0.x - r.o.x
                qy = tri.p0.y - r.o.y
                qz = tri.p0.z - r.o.z
                num = qx * wx + qy * wy + qz * wz  # TegVar-free

                # The ray hits the plane
                t = num / denom  # Teg-free

                # Heaviside(HitDiffeo([self.p0.x], []))

                # The hit never happens because time >= 0
                # if t < 0:
                #     return Hit3(r, None)

                # The ray to the plane is Ray3(r.o, k * r.d)
                # Is that point in the triangle?
                ppdx = r.o.x * denom + num * r.d.x - tri.p0.x * denom  # Linear in TegVar
                ppdy = r.o.y * denom + num * r.d.y - tri.p0.y * denom  # Linear in TegVar
                ppdz = r.o.z * denom + num * r.d.z - tri.p0.z * denom  # Linear in TegVar
                ppd = Vec3(ppdx, ppdy, ppdz)

                denom2 = u.dot(v) ** 2 - u.dot(u) * v.dot(v)  # TegVar-free

                si = (u.dot(v) * ppd.dot(v) - v.dot(v) * ppd.dot(u)) / denom2
                si = simplify(si)
                # si = (u.dot(v) * ppd.dot(v) - v.dot(v) * ppd.dot(u)) / (denom2 * denom)
                # (si < 0.0)
                vars = (VAR_S, VAR_T, pix_x, pix_y)
                tvars = (deltax, deltay)
                denom_diffeo = make_diffeo(denom, vars, tvars)
                minus_denom_diffeo = make_diffeo(-denom, vars, tvars)
                denom_diffeo_gt0 = Heaviside(denom_diffeo)
                denom_diffeo_lt0 = Heaviside(minus_denom_diffeo)

                si_gt0 = Heaviside(make_diffeo(si, vars, tvars))
                si_lt0 = Heaviside(make_diffeo(-si, vars, tvars))
                si_denom_gt0 = denom_diffeo_gt0 * si_gt0 + denom_diffeo_lt0 * si_lt0

                ti = (u.dot(v) * ppd.dot(u) - u.dot(u) * ppd.dot(v)) / denom2
                ti = simplify(ti)
                # ti = (u.dot(v) * ppd.dot(u) - u.dot(u) * ppd.dot(v)) / (denom2 * denom)
                # (ti < 0.0)
                ti_gt0 = Heaviside(make_diffeo(ti, vars, tvars))
                ti_lt0 = Heaviside(make_diffeo(-ti, vars, tvars))
                ti_denom_gt0 = denom_diffeo_gt0 * ti_gt0 + denom_diffeo_lt0 * ti_lt0

                # si + ti < 1.0
                si_ti_lt1 = Heaviside(make_diffeo(denom - si - ti, vars, tvars))
                si_ti_gt1 = Heaviside(make_diffeo(si + ti - denom, vars, tvars))
                sipti_lt1 = denom_diffeo_gt0 * si_ti_lt1 + denom_diffeo_lt0 * si_ti_gt1
                w = denom_diffeo.weight.right.expr
                for _ in range(100):
                    w = simplify(
                        w,
                        VarVal({VAR_T.name: 1, VAR_S.name: -20}),
                    )

                f = denom_diffeo.function(denom_diffeo.vars, denom_diffeo.tvars)[0]
                for _ in range(100):
                    f = simplify(
                        f,
                        VarVal({VAR_T.name: 1, VAR_S.name: -20}),
                    )

                t_gt0 = Heaviside(make_diffeo(t, vars, tvars))

                all_hits.append(
                    (
                        Hit3(r, t),
                        si_denom_gt0 * ti_denom_gt0 * sipti_lt1 * t_gt0,
                        Vec3(wx, wy, wz),
                    )
                )
        hit: Hit3 = all_hits[0][0]
        does_hit: GExpr = all_hits[0][1]
        n: Vec3 = all_hits[0][2]
        var_val = VarVal({VAR_T.name: 1, VAR_S.name: -20, pix_x.name: 0, pix_y.name: 0})
        does_hit = simplify(does_hit, var_val)
        threshold = 0.3
        sample_color = Colors.RED
        black_color = Colors.BLACK
        toon_color = Colors.BLUE
        val = n.x * ray.d.x + n.y * ray.d.y + n.z * ray.d.z
        val_gt0 = Heaviside(make_diffeo(val, vars, tvars))
        val_gt0_compare_gt0 = Heaviside(make_diffeo(threshold - val, vars, tvars))
        val_lt0_compare_gt0 = Heaviside(make_diffeo(threshold + val, vars, tvars))
        val_lt_threshold = val_gt0 * val_gt0_compare_gt0 + (-val_gt0 * val_lt0_compare_gt0)
        final_color_r = black_color.r * (1 - does_hit) + sample_color.r * does_hit * (1 - val_lt_threshold) + toon_color.r * does_hit * val_lt_threshold
        final_color_g = black_color.g * (1 - does_hit) + sample_color.g * does_hit * (1 - val_lt_threshold) + toon_color.g * does_hit * val_lt_threshold
        final_color_b = black_color.b * (1 - does_hit) + sample_color.b * does_hit * (1 - val_lt_threshold) + toon_color.b * does_hit * val_lt_threshold
        return [final_color_r, final_color_g, final_color_b, Const(1)]

    def ray_trace_xy(self, var_val: VarVal, intr: GExpr, intg: GExpr, intb: GExpr, is_primal: bool) -> list[float]:
        num_samples = self.num_samples

        sym_dt = Sym("dt")
        sym_ds = Sym("ds")
        derive_ctx = {k: k for k in var_val} | {VAR_T.name: sym_dt, VAR_S.name: sym_ds}
        pix_color = {}
        for i, intx in enumerate((intr, intg, intb)):
            intx = simplify(intx, var_val)

            if is_primal:
                expr_dt = intx
            else:
                expr_dt = deriv(intx, derive_ctx)

            var_val = VarVal({VAR_T.name: 1, VAR_S.name: -20, sym_dt: 1, sym_ds: 1} | var_val._data)

            expr_dt = simplify(expr_dt, var_val)

            pix_color[i] = evaluate_all(expr_dt, var_val, num_samples=num_samples)
        pix_color = Color(pix_color[0] * num_samples, pix_color[1]  * num_samples, pix_color[2]  * num_samples, num_samples)
        return (pix_color).tolist()

    def build_diff_rt_expr(self, pix_x, pix_y, screen_width, screen_height) -> tuple[GExpr, GExpr, GExpr]:
        cam = self.camera

        def f(deltax, deltay):
            ndc = cam.screen_point_to_ndc(Pnt2(pix_x + deltax, pix_y + deltay), screen_width, screen_height)
            cam_to_point = cam.diff_ndc_to_ray(ndc)
            return self.radiance(cam_to_point, pix_x, pix_y)

        mux = BoundedLebesgue(0, 1, deltax)
        nuy = BoundedLebesgue(0, 1, deltay)
        r, g, b, _ = f(deltax, deltay)
        intr = Int(Int(r, mux), nuy)
        intg = Int(Int(g, mux), nuy)
        intb = Int(Int(b, mux), nuy)
        return (intr, intg, intb)

    def ray_trace(self, screen_width, screen_height, is_primal):
        c = np.zeros((screen_width, screen_height, 4))
        intr, intg, intb = self.build_diff_rt_expr(pix_x, pix_y, screen_width, screen_height)
        for x in range(screen_width):
            for y in range(screen_height):
                var_val = VarVal({pix_x.name: x, pix_y.name: y})
                c[x, y] = self.ray_trace_xy(var_val, intr, intg, intb, is_primal)
                print(f"traced {x, y} {c[x, y]}")
        return c

    def ray_trace_parallel(self, screen_width, screen_height, is_primal):
        intr, intg, intb = self.build_diff_rt_expr(pix_x, pix_y, screen_width, screen_height)
        def ray_trace_xy_proxy(x, y):
            var_val = VarVal({pix_x.name: x, pix_y.name: y})
            c = self.ray_trace_xy(var_val, intr, intg, intb, is_primal)
            print((x, y), c)
            return (x, y), c

        results = Parallel(n_jobs=cpu_count() - 1)(
            delayed(ray_trace_xy_proxy)(x, y) for x in range(screen_width) for y in range(screen_height)
        )

        c = np.zeros((screen_width, screen_height, 4))
        for k, v in results:
            c[k] = v
        return c

@dataclass
class RayTraceThresholdLambertianPotto(RayTracePotto):
    def radiance(self, r: Ray3, pix_x: Var, pix_y: Var, depth: int = 0) -> Color:
        ray = r
        vars = (pix_x, pix_y)
        tvars = (deltax, deltay)
        all_hits = []
        for n in self.scene.nodes:
            for shape in n.shapes:
                tri: Triangle = shape
                nt = n.transform.inverse()
                v = Vec3(r.o.x, r.o.y, r.o.z)
                matmulofro = Pnt3(nt.rot_scale.x0 * v, nt.rot_scale.x1 * v, nt.rot_scale.x2 * v)
                newro = matmulofro + nt.translate

                v = Vec3(r.d.x, r.d.y, r.d.z)
                matmulofrd = Pnt3(nt.rot_scale.x0 * v, nt.rot_scale.x1 * v, nt.rot_scale.x2 * v)
                r = Ray3(newro, matmulofrd)

                u = tri.p1 - tri.p0
                v = tri.p2 - tri.p0

                wx = u.y * v.z - u.z * v.y
                wy = u.z * v.x - u.x * v.z
                wz = u.x * v.y - u.y * v.x

                denom = r.d.x * wx + r.d.y * wy + r.d.z * wz  # Linear in TegVars
                denom = simplify(denom)

                qx = tri.p0.x - r.o.x
                qy = tri.p0.y - r.o.y
                qz = tri.p0.z - r.o.z
                num = qx * wx + qy * wy + qz * wz  # TegVar-free

                # The ray hits the plane
                t = num / denom  # Teg-free

                # Heaviside(HitDiffeo([self.p0.x], []))

                # The hit never happens because time >= 0
                # if t < 0:
                #     return Hit3(r, None)

                # The ray to the plane is Ray3(r.o, k * r.d)
                # Is that point in the triangle?
                ppdx = r.o.x * denom + num * r.d.x - tri.p0.x * denom  # Linear in TegVar
                ppdy = r.o.y * denom + num * r.d.y - tri.p0.y * denom  # Linear in TegVar
                ppdz = r.o.z * denom + num * r.d.z - tri.p0.z * denom  # Linear in TegVar
                ppd = Vec3(ppdx, ppdy, ppdz)

                denom2 = u.dot(v) ** 2 - u.dot(u) * v.dot(v)  # TegVar-free

                si = (u.dot(v) * ppd.dot(v) - v.dot(v) * ppd.dot(u)) / denom2
                si = simplify(si)
                # si = (u.dot(v) * ppd.dot(v) - v.dot(v) * ppd.dot(u)) / (denom2 * denom)
                # (si < 0.0)
                # vars = (VAR_S, VAR_T, pix_x, pix_y)
                vars = (VAR_T,)
                tvars = (deltax, deltay)
                denom_diffeo = make_diffeo(denom, vars, tvars)
                minus_denom_diffeo = make_diffeo(-denom, vars, tvars)
                denom_diffeo_gt0 = Heaviside(denom_diffeo)
                denom_diffeo_lt0 = Heaviside(minus_denom_diffeo)

                si_gt0 = Heaviside(make_diffeo(si, vars, tvars))
                si_lt0 = Heaviside(make_diffeo(-si, vars, tvars))
                si_denom_gt0 = denom_diffeo_gt0 * si_gt0 + denom_diffeo_lt0 * si_lt0

                ti = (u.dot(v) * ppd.dot(u) - u.dot(u) * ppd.dot(v)) / denom2
                ti = simplify(ti)
                # ti = (u.dot(v) * ppd.dot(u) - u.dot(u) * ppd.dot(v)) / (denom2 * denom)
                # (ti < 0.0)
                ti_gt0 = Heaviside(make_diffeo(ti, vars, tvars))
                ti_lt0 = Heaviside(make_diffeo(-ti, vars, tvars))
                ti_denom_gt0 = denom_diffeo_gt0 * ti_gt0 + denom_diffeo_lt0 * ti_lt0

                # si + ti < 1.0
                si_ti_lt1 = Heaviside(make_diffeo(denom - si - ti, vars, tvars))
                si_ti_gt1 = Heaviside(make_diffeo(si + ti - denom, vars, tvars))
                sipti_lt1 = denom_diffeo_gt0 * si_ti_lt1 + denom_diffeo_lt0 * si_ti_gt1
                w = denom_diffeo.weight.right.expr
                for _ in range(100):
                    w = simplify(
                        w,
                        VarVal({VAR_T.name: 1, VAR_S.name: -20}),
                    )

                f = denom_diffeo.function(denom_diffeo.vars, denom_diffeo.tvars)[0]
                for _ in range(100):
                    f = simplify(
                        f,
                        VarVal({VAR_T.name: 1, VAR_S.name: -20}),
                    )

                t_gt0 = Heaviside(make_diffeo(t, vars, tvars))

                all_hits.append(
                    (
                        Hit3(r, t),
                        si_denom_gt0 * ti_denom_gt0 * sipti_lt1 * t_gt0,
                        Vec3(wx, wy, wz),
                    )
                )
        hit: Hit3 = all_hits[0][0]
        does_hit: GExpr = all_hits[0][1]
        n: Vec3 = all_hits[0][2]
        var_val = VarVal({VAR_T.name: 1, VAR_S.name: -20, pix_x.name: 0, pix_y.name: 0})
        does_hit = simplify(does_hit, var_val)
        sample_color = Colors.GREEN
        # hack
        light: PointLight = self.scene.lights[0]
        light_color = light.get_color()
        light_pos = light.pos
        light_dir_x = light_pos.x - (ray.o.x + hit.t * ray.d.x)
        light_dir_y = light_pos.y - (ray.o.y + hit.t * ray.d.y)
        light_dir_z = light_pos.z - (ray.o.z + hit.t * ray.d.z)
        inv_factor = light_dir_x * light_dir_x + light_dir_y * light_dir_y + light_dir_z * light_dir_z
        hit_color = sample_color * light_color / (inv_factor * M_PI)
        threshold = 0.005
        g_gt_thrseshold = Heaviside(make_diffeo(hit_color.g - threshold, vars, tvars))
        n_x = 0.06226324077979757
        n_y = 0.062319272093368024
        n_z = 0.9961122412526359
        cos_theta = n_x * light_dir_x + n_y * light_dir_y + n_z * light_dir_z
        black_color = Colors.BLACK
        final_color_r = hit_color.r * cos_theta
        final_color_g = hit_color.g * cos_theta
        final_color_b = hit_color.b * cos_theta
        output_color_r = final_color_r * g_gt_thrseshold * does_hit + black_color.r * (1 - does_hit) + Colors.RED.r * does_hit * (1 - g_gt_thrseshold)
        output_color_g = final_color_g * g_gt_thrseshold * does_hit + black_color.g * (1 - does_hit) + Colors.RED.g * does_hit * (1 - g_gt_thrseshold)
        output_color_b = final_color_b * g_gt_thrseshold * does_hit + black_color.b * (1 - does_hit) + Colors.RED.b * does_hit * (1 - g_gt_thrseshold)
        return [output_color_r, output_color_g, output_color_b, Const(1)]


@dataclass
class RayTraceSwappablePotto(RayTrace):
    """
    Potto Shader Implementation
    """
    def radiance_toon(self, deltax: TegVar, deltay: TegVar, pix_x: Var, pix_y: Var, depth: int = 0) -> Color:
        cam = self.camera
        ndc = cam.screen_point_to_ndc(Pnt2(pix_x + deltax, pix_y + deltay), self.cur_screen_width, self.cur_screen_height)
        cam_to_point = cam.diff_ndc_to_ray(ndc)
        r = cam_to_point
        ray = r
        vars = (pix_x, pix_y)
        tvars = (deltax, deltay)
        all_hits = []
        for n in self.scene.nodes:
            for shape in n.shapes:
                tri: Triangle = shape
                nt = n.transform.inverse()
                v = Vec3(r.o.x, r.o.y, r.o.z)
                matmulofro = Pnt3(nt.rot_scale.x0 * v, nt.rot_scale.x1 * v, nt.rot_scale.x2 * v)
                newro = matmulofro + nt.translate

                v = Vec3(r.d.x, r.d.y, r.d.z)
                matmulofrd = Pnt3(nt.rot_scale.x0 * v, nt.rot_scale.x1 * v, nt.rot_scale.x2 * v)
                r = Ray3(newro, matmulofrd)

                u = tri.p1 - tri.p0
                v = tri.p2 - tri.p0

                wx = u.y * v.z - u.z * v.y
                wy = u.z * v.x - u.x * v.z
                wz = u.x * v.y - u.y * v.x

                denom = r.d.x * wx + r.d.y * wy + r.d.z * wz  # Linear in TegVars
                denom = simplify(denom)

                qx = tri.p0.x - r.o.x
                qy = tri.p0.y - r.o.y
                qz = tri.p0.z - r.o.z
                num = qx * wx + qy * wy + qz * wz  # TegVar-free

                # The ray hits the plane
                t = num / denom  # Teg-free

                # Heaviside(HitDiffeo([self.p0.x], []))

                # The hit never happens because time >= 0
                # if t < 0:
                #     return Hit3(r, None)

                # The ray to the plane is Ray3(r.o, k * r.d)
                # Is that point in the triangle?
                ppdx = r.o.x * denom + num * r.d.x - tri.p0.x * denom  # Linear in TegVar
                ppdy = r.o.y * denom + num * r.d.y - tri.p0.y * denom  # Linear in TegVar
                ppdz = r.o.z * denom + num * r.d.z - tri.p0.z * denom  # Linear in TegVar
                ppd = Vec3(ppdx, ppdy, ppdz)

                denom2 = u.dot(v) ** 2 - u.dot(u) * v.dot(v)  # TegVar-free

                si = (u.dot(v) * ppd.dot(v) - v.dot(v) * ppd.dot(u)) / denom2
                si = simplify(si)
                # si = (u.dot(v) * ppd.dot(v) - v.dot(v) * ppd.dot(u)) / (denom2 * denom)
                # (si < 0.0)
                vars = (VAR_S, VAR_T, pix_x, pix_y)
                tvars = (deltax, deltay)
                denom_diffeo = make_diffeo(denom, vars, tvars)
                minus_denom_diffeo = make_diffeo(-denom, vars, tvars)
                denom_diffeo_gt0 = Heaviside(denom_diffeo)
                denom_diffeo_lt0 = Heaviside(minus_denom_diffeo)

                si_gt0 = Heaviside(make_diffeo(si, vars, tvars))
                si_lt0 = Heaviside(make_diffeo(-si, vars, tvars))
                si_denom_gt0 = denom_diffeo_gt0 * si_gt0 + denom_diffeo_lt0 * si_lt0

                ti = (u.dot(v) * ppd.dot(u) - u.dot(u) * ppd.dot(v)) / denom2
                ti = simplify(ti)
                # ti = (u.dot(v) * ppd.dot(u) - u.dot(u) * ppd.dot(v)) / (denom2 * denom)
                # (ti < 0.0)
                ti_gt0 = Heaviside(make_diffeo(ti, vars, tvars))
                ti_lt0 = Heaviside(make_diffeo(-ti, vars, tvars))
                ti_denom_gt0 = denom_diffeo_gt0 * ti_gt0 + denom_diffeo_lt0 * ti_lt0

                # si + ti < 1.0
                si_ti_lt1 = Heaviside(make_diffeo(denom - si - ti, vars, tvars))
                si_ti_gt1 = Heaviside(make_diffeo(si + ti - denom, vars, tvars))
                sipti_lt1 = denom_diffeo_gt0 * si_ti_lt1 + denom_diffeo_lt0 * si_ti_gt1
                w = denom_diffeo.weight.right.expr
                for _ in range(100):
                    w = simplify(
                        w,
                        VarVal({VAR_T.name: 1, VAR_S.name: -20}),
                    )

                f = denom_diffeo.function(denom_diffeo.vars, denom_diffeo.tvars)[0]
                for _ in range(100):
                    f = simplify(
                        f,
                        VarVal({VAR_T.name: 1, VAR_S.name: -20}),
                    )

                t_gt0 = Heaviside(make_diffeo(t, vars, tvars))

                all_hits.append(
                    (
                        Hit3(r, t),
                        si_denom_gt0 * ti_denom_gt0 * sipti_lt1 * t_gt0,
                        Vec3(wx, wy, wz),
                    )
                )
        hit: Hit3 = all_hits[0][0]
        does_hit: GExpr = all_hits[0][1]
        n: Vec3 = all_hits[0][2]
        var_val = VarVal({VAR_T.name: 1, VAR_S.name: -20, pix_x.name: 0, pix_y.name: 0})
        does_hit = simplify(does_hit, var_val)
        threshold = 0.3
        sample_color = Colors.RED
        black_color = Colors.BLACK
        toon_color = Colors.BLUE
        val = n.x * ray.d.x + n.y * ray.d.y + n.z * ray.d.z
        val_gt0 = Heaviside(make_diffeo(val, vars, tvars))
        val_gt0_compare_gt0 = Heaviside(make_diffeo(threshold - val, vars, tvars))
        val_lt0_compare_gt0 = Heaviside(make_diffeo(threshold + val, vars, tvars))
        val_lt_threshold = val_gt0 * val_gt0_compare_gt0 + (-val_gt0 * val_lt0_compare_gt0)
        final_color_r = black_color.r * (1 - does_hit) + sample_color.r * does_hit * (1 - val_lt_threshold) + toon_color.r * does_hit * val_lt_threshold
        final_color_g = black_color.g * (1 - does_hit) + sample_color.g * does_hit * (1 - val_lt_threshold) + toon_color.g * does_hit * val_lt_threshold
        final_color_b = black_color.b * (1 - does_hit) + sample_color.b * does_hit * (1 - val_lt_threshold) + toon_color.b * does_hit * val_lt_threshold
        return [final_color_r, final_color_g, final_color_b, Const(1)]

    def radiance_lambertian(self, deltax: TegVar, deltay: TegVar, pix_x: Var, pix_y: Var, depth: int = 0) -> Color:
        cam = self.camera
        ndc = cam.screen_point_to_ndc(Pnt2(pix_x + deltax, pix_y + deltay), self.cur_screen_width, self.cur_screen_height)
        cam_to_point = cam.diff_ndc_to_ray(ndc)
        r = cam_to_point
        ray = r
        vars = (pix_x, pix_y)
        tvars = (deltax, deltay)
        all_hits = []
        for n in self.scene.nodes:
            for shape in n.shapes:
                tri: Triangle = shape
                nt = n.transform.inverse()
                v = Vec3(r.o.x, r.o.y, r.o.z)
                matmulofro = Pnt3(nt.rot_scale.x0 * v, nt.rot_scale.x1 * v, nt.rot_scale.x2 * v)
                newro = matmulofro + nt.translate

                v = Vec3(r.d.x, r.d.y, r.d.z)
                matmulofrd = Pnt3(nt.rot_scale.x0 * v, nt.rot_scale.x1 * v, nt.rot_scale.x2 * v)
                r = Ray3(newro, matmulofrd)

                u = tri.p1 - tri.p0
                v = tri.p2 - tri.p0

                wx = u.y * v.z - u.z * v.y
                wy = u.z * v.x - u.x * v.z
                wz = u.x * v.y - u.y * v.x

                denom = r.d.x * wx + r.d.y * wy + r.d.z * wz  # Linear in TegVars
                denom = simplify(denom)

                qx = tri.p0.x - r.o.x
                qy = tri.p0.y - r.o.y
                qz = tri.p0.z - r.o.z
                num = qx * wx + qy * wy + qz * wz  # TegVar-free

                # The ray hits the plane
                t = num / denom  # Teg-free

                # Heaviside(HitDiffeo([self.p0.x], []))

                # The hit never happens because time >= 0
                # if t < 0:
                #     return Hit3(r, None)

                # The ray to the plane is Ray3(r.o, k * r.d)
                # Is that point in the triangle?
                ppdx = r.o.x * denom + num * r.d.x - tri.p0.x * denom  # Linear in TegVar
                ppdy = r.o.y * denom + num * r.d.y - tri.p0.y * denom  # Linear in TegVar
                ppdz = r.o.z * denom + num * r.d.z - tri.p0.z * denom  # Linear in TegVar
                ppd = Vec3(ppdx, ppdy, ppdz)

                denom2 = u.dot(v) ** 2 - u.dot(u) * v.dot(v)  # TegVar-free

                si = (u.dot(v) * ppd.dot(v) - v.dot(v) * ppd.dot(u)) / denom2
                si = simplify(si)
                # si = (u.dot(v) * ppd.dot(v) - v.dot(v) * ppd.dot(u)) / (denom2 * denom)
                # (si < 0.0)
                vars = (VAR_S, VAR_T, pix_x, pix_y)
                tvars = (deltax, deltay)
                denom_diffeo = make_diffeo(denom, vars, tvars)
                minus_denom_diffeo = make_diffeo(-denom, vars, tvars)
                denom_diffeo_gt0 = Heaviside(denom_diffeo)
                denom_diffeo_lt0 = Heaviside(minus_denom_diffeo)

                si_gt0 = Heaviside(make_diffeo(si, vars, tvars))
                si_lt0 = Heaviside(make_diffeo(-si, vars, tvars))
                si_denom_gt0 = denom_diffeo_gt0 * si_gt0 + denom_diffeo_lt0 * si_lt0

                ti = (u.dot(v) * ppd.dot(u) - u.dot(u) * ppd.dot(v)) / denom2
                ti = simplify(ti)
                # ti = (u.dot(v) * ppd.dot(u) - u.dot(u) * ppd.dot(v)) / (denom2 * denom)
                # (ti < 0.0)
                ti_gt0 = Heaviside(make_diffeo(ti, vars, tvars))
                ti_lt0 = Heaviside(make_diffeo(-ti, vars, tvars))
                ti_denom_gt0 = denom_diffeo_gt0 * ti_gt0 + denom_diffeo_lt0 * ti_lt0

                # si + ti < 1.0
                si_ti_lt1 = Heaviside(make_diffeo(denom - si - ti, vars, tvars))
                si_ti_gt1 = Heaviside(make_diffeo(si + ti - denom, vars, tvars))
                sipti_lt1 = denom_diffeo_gt0 * si_ti_lt1 + denom_diffeo_lt0 * si_ti_gt1
                w = denom_diffeo.weight.right.expr
                for _ in range(100):
                    w = simplify(
                        w,
                        VarVal({VAR_T.name: 1, VAR_S.name: -20}),
                    )

                f = denom_diffeo.function(denom_diffeo.vars, denom_diffeo.tvars)[0]
                for _ in range(100):
                    f = simplify(
                        f,
                        VarVal({VAR_T.name: 1, VAR_S.name: -20}),
                    )

                t_gt0 = Heaviside(make_diffeo(t, vars, tvars))

                all_hits.append(
                    (
                        Hit3(r, t),
                        si_denom_gt0 * ti_denom_gt0 * sipti_lt1 * t_gt0,
                        Vec3(wx, wy, wz),
                    )
                )
        hit: Hit3 = all_hits[0][0]
        does_hit: GExpr = all_hits[0][1]
        n: Vec3 = all_hits[0][2]
        var_val = VarVal({VAR_T.name: 1, VAR_S.name: -20, pix_x.name: 0, pix_y.name: 0})
        does_hit = simplify(does_hit, var_val)
        threshold = 0.3
        sample_color = Colors.GREEN
        # hack
        light: PointLight = self.scene.lights[0]
        light_color = light.get_color()
        light_pos = light.pos
        light_dir_x = light_pos.x - (ray.o.x + hit.t * ray.d.x)
        light_dir_y = light_pos.y - (ray.o.y + hit.t * ray.d.y)
        light_dir_z = light_pos.z - (ray.o.z + hit.t * ray.d.z)
        light_dir_norm = Sqrt(light_dir_x * light_dir_x + light_dir_y * light_dir_y + light_dir_z * light_dir_z)
        light_dir_x = light_dir_x / light_dir_norm
        light_dir_y = light_dir_y / light_dir_norm
        light_dir_z = light_dir_z / light_dir_norm

        final_color = sample_color * light_color
        n_norm = Sqrt(n.x * n.x + n.y * n.y + n.z * n.z)
        n_x = n.x / n_norm
        n_y = n.y / n_norm
        n_z = n.z / n_norm
        cos_theta = n_x * light_dir_x + n_y * light_dir_y + n_z * light_dir_z
        black_color = Colors.BLACK
        final_color_r = black_color.r * (1 - does_hit) + final_color.r * cos_theta / M_PI * light_color.r * does_hit
        final_color_g = black_color.g * (1 - does_hit) + final_color.g * cos_theta / M_PI * light_color.r * does_hit
        final_color_b = black_color.b * (1 - does_hit) + final_color.b * cos_theta / M_PI * light_color.r * does_hit
        return [final_color_r, final_color_g, final_color_b, Const(1)]

    def ray_trace_xy(self, var_val: VarVal, intr: GExpr, intg: GExpr, intb: GExpr) -> list[float]:
        num_samples = self.num_samples

        sym_dt = Sym("dt")
        sym_ds = Sym("ds")
        derive_ctx = {k: k for k in var_val} | {VAR_T.name: sym_dt, VAR_S.name: sym_ds}
        pix_color = {}
        for i, intx in enumerate((intr, intg, intb)):
            # print(f"Simplifying {i}")
            intx = simplify(intx, var_val)

            # print("Starting deriv")
            expr_dt = deriv(intx, derive_ctx)
            # print("derivative computed")

            var_val = VarVal({VAR_T.name: 1, VAR_S.name: -20, sym_dt: 1, sym_ds: 1} | var_val._data)
            # print("Simplifying the derivative expression")
            expr_dt = simplify(expr_dt, var_val)

            pix_color[i] = evaluate_all(intx, env_or_var_val=var_val, num_samples=num_samples)
        pix_color = Color(pix_color[0] * num_samples, pix_color[1]  * num_samples, pix_color[2]  * num_samples, num_samples)
        return (pix_color).tolist()

    def build_diff_rt_expr_swappable(self, pix_x, pix_y, screen_width, screen_height, is_default_shader = True) -> tuple[GExpr, GExpr, GExpr]:
        self.cur_screen_width = screen_width
        self.cur_screen_height = screen_height
        # cam = self.camera
        # ndc = cam.screen_point_to_ndc(Pnt2(pix_x + deltax, pix_y + deltay), screen_width, screen_height)
        # cam_to_point = cam.diff_ndc_to_ray(ndc)

        # def f_toon(deltax, deltay):
        #     ndc = cam.screen_point_to_ndc(Pnt2(pix_x + deltax, pix_y + deltay), screen_width, screen_height)
        #     cam_to_point = cam.diff_ndc_to_ray(ndc)
        #     return self.radiance_toon(deltax, deltay, pix_x, pix_y)

        # def f_lambertian(deltax, deltay):
        #     ndc = cam.screen_point_to_ndc(Pnt2(pix_x + deltax, pix_y + deltay), screen_width, screen_height)
        #     cam_to_point = cam.diff_ndc_to_ray(ndc)
        #     return self.radiance_lambertian(deltax, deltay, pix_x, pix_y)


        shader = Var("shader")
        dx, dy = TegVar("dx"), TegVar("dy")
        px, py = Var("px"), Var("py")
        toon_f_r = Function((dx, dy, px, py), self.radiance_toon(dx, dy, px, py)[0], Sym("toon_f_r"))
        toon_f_g = Function((dx, dy, px, py), self.radiance_toon(dx, dy, px, py)[1], Sym("toon_f_g"))
        toon_f_b = Function((dx, dy, px, py), self.radiance_toon(dx, dy, px, py)[2], Sym("toon_f_b"))

        lambertian_f_r = Function((dx, dy, px, py), self.radiance_lambertian(dx, dy, px, py)[0], Sym("lambertian_f_r"))
        lambertian_f_g = Function((dx, dy, px, py), self.radiance_lambertian(dx, dy, px, py)[1], Sym("lambertian_f_g"))
        lambertian_f_b = Function((dx, dy, px, py), self.radiance_lambertian(dx, dy, px, py)[2], Sym("lambertian_f_b"))

        # define higher order function that takes a shader function
        f_shader = Function((shader,), App(shader, (deltax, deltay, pix_x, pix_y), "app_shader"), Sym("f_shader"))
        # apply shader function to the higher order function to get the final experession
        if is_default_shader:
            r = App(f_shader, (toon_f_r,), "t_r_app")
            g = App(f_shader, (toon_f_g,), "t_g_app")
            b = App(f_shader, (toon_f_b,), "t_b_app")
        else:
            r = App(f_shader, (lambertian_f_r,), "l_r_app")
            g = App(f_shader, (lambertian_f_g,), "l_g_app")
            b = App(f_shader, (lambertian_f_b,), "l_b_app")

        mux = BoundedLebesgue(0, 1, deltax)
        nuy = BoundedLebesgue(0, 1, deltay)
        intr = Int(Int(r, mux), nuy)
        intg = Int(Int(g, mux), nuy)
        intb = Int(Int(b, mux), nuy)
        return (intr, intg, intb)

    def ray_trace(self, screen_width, screen_height, is_default_shader):
        c = np.zeros((screen_width, screen_height, 4))
        intr, intg, intb = self.build_diff_rt_expr_swappable(pix_x, pix_y, screen_width, screen_height, is_default_shader)
        for x in range(screen_width):
            for y in range(screen_height):
                var_val = VarVal({pix_x.name: x, pix_y.name: y})
                c[x, y] = self.ray_trace_xy(var_val, intr, intg, intb)
                print(f"traced {x, y} {c[x, y]}")
        return c

    def ray_trace_parallel(self, screen_width, screen_height, is_default_shader):
        intr, intg, intb = self.build_diff_rt_expr_swappable(pix_x, pix_y, screen_width, screen_height, is_default_shader)
        def ray_trace_xy_proxy(x, y):
            var_val = VarVal({pix_x.name: x, pix_y.name: y})
            c = self.ray_trace_xy(var_val, intr, intg, intb)
            print((x, y), c)
            return (x, y), c

        results = Parallel(n_jobs=cpu_count() - 1)(
            delayed(ray_trace_xy_proxy)(x, y) for x in range(screen_width) for y in range(screen_height)
        )

        c = np.zeros((screen_width, screen_height, 4))
        for k, v in results:
            c[k] = v
        return c
