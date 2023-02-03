import numpy as np
import matplotlib.pyplot as plt

from graphics.scene.scenegraph import SceneGraph, SceneNode
from graphics.scene.raytrace import RayTrace, RayTraceThresholdLambertianPotto, RayTraceLambertianPotto, RayTraceDirectPython, RayTraceDirectPythonThreshold, RayTracePotto, RayTracePython, RayTraceSwappablePotto
from graphics.scene.camera import TurntableCamera

from graphics.util.color import Colors
from graphics.util.mat3 import Tform3
from graphics.util.vec3 import Pnt3, Vec3
from graphics.scene.shape3 import Triangle, VAR_T, VAR_VAL
from graphics.scene.light import PointLight

from potto import Const
from potto import evaluate_all
import time

IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16

def main():
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
                        Pnt3(Const(8), Const(-8), Const(0 - 1e-4)),
                        Pnt3(Const(-8), Const(8), Const(-1e-3)),
                        e=Colors.WHITE,
                    ),
                ],
            ),
        ],
        [
            PointLight(
                to_raw_pnt(VAR_VAL, Pnt3(Const(0), Const(0), Const(1.0))),
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
        ],
        [
            PointLight(
                to_raw_pnt(VAR_VAL, Pnt3(Const(0), Const(0), Const(1.0))),
                intensity=Colors.WHITE,
            )
        ]
    )

    # slanted triangle
    slanted_sg = SceneGraph(
        [
            SceneNode(
                transform=Tform3(Vec3(0, 0, 0)),
                shapes=[
                    Triangle(
                        Pnt3(Const(-8), Const(-8), Const(1.0)),
                        Pnt3(Const(8), Const(-8), Const(-1.0 + 1e-3)),
                        Pnt3(Const(-8), Const(8), Const(-1.0 + 1e-4)),
                        e=Colors.WHITE,
                    ),
                ],
            ),
        ],
        [
            PointLight(
                to_raw_pnt(VAR_VAL, Pnt3(Const(0), Const(0), Const(1.5))),
                intensity=Colors.WHITE,
            )
        ]
    )

    slanted_raw_sg = SceneGraph(
        [
            SceneNode(
                transform=Tform3(Vec3(0, 0, 0)),
                shapes=[
                    Triangle(
                        to_raw_pnt(VAR_VAL, Pnt3(Const(-8), Const(-8), Const(1.0))),
                        to_raw_pnt(VAR_VAL, Pnt3(Const(8), Const(-8), Const(-1.0))),
                        to_raw_pnt(VAR_VAL, Pnt3(Const(-8), Const(8), Const(-1.0))),
                        e=Colors.WHITE,
                    ),
                ],
            ),
        ],
        [
            PointLight(
                to_raw_pnt(VAR_VAL, Pnt3(Const(0), Const(0), Const(1.5))),
                intensity=Colors.WHITE,
            )
        ]
    )

    aspect_ratio = IMAGE_WIDTH / IMAGE_HEIGHT
    cam = TurntableCamera(dist=7, angle=0, aspect_ratio=aspect_ratio, fov_angle_y=90*np.pi/180)
    rt = RayTrace(sg, cam, 10)
    rt2 = RayTracePython(raw_sg, cam, 10)
    rt3 = RayTracePotto(sg, cam, 10)
    rt4 = RayTraceDirectPython(raw_sg, cam, 10)
    rt5 = RayTraceLambertianPotto(sg, cam, 10)
    rt6 = RayTraceSwappablePotto(sg, cam, 10)
    rt7 = RayTraceDirectPythonThreshold(slanted_raw_sg, cam, 10)
    rt8 = RayTraceThresholdLambertianPotto(slanted_sg, cam, 10)

    toon_shader_img = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 4))
    lambertian_shader_img = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 4))
    ref_img = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, 4))

    start = time.time()
    potto_img = rt8.ray_trace_toon_parallel(IMAGE_WIDTH, IMAGE_HEIGHT)
    # potto_img = rt7.ray_trace_toon_parallel_hack(IMAGE_WIDTH, IMAGE_HEIGHT)
    end = time.time()
    print(f"rendering took {end - start} seconds")

    plt.axis('off')
    plt.imshow(potto_img)
    plt.show()


if __name__ == "__main__":
    main()
