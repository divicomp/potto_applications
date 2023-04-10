import numpy as np
import matplotlib.pyplot as plt

from graphics.scene.scenegraph import SceneGraph, SceneNode
from graphics.scene.raytrace import RayTraceThresholdLambertianPotto
from graphics.scene.camera import TurntableCamera

from graphics.util.color import Colors
from graphics.util.mat3 import Tform3
from graphics.util.vec3 import Pnt3, Vec3
from graphics.scene.shape3 import Triangle, VAR_T, VAR_VAL
from graphics.scene.light import PointLight

from potto import Const
from potto import evaluate_all
import time
import argparse


def main(image_width, image_height, is_primal):
    def to_raw_pnt(var_vals, p: Pnt3):
        def ev(e):
            return evaluate_all(e, var_vals)
        return Pnt3(ev(p.x), ev(p.y), ev(p.z))

    light_sg = SceneGraph(
        [
            SceneNode(
                transform=Tform3(Vec3(0, 0, 0)),
                shapes=[
                    Triangle(
                        Pnt3(Const(-8), Const(-8), Const(1.0) * VAR_T),
                        Pnt3(Const(8) * VAR_T, Const(-8), Const(0 - 1e-4)),
                        Pnt3(Const(-8), Const(8)  * VAR_T, Const(-1e-3)),
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

    raw_light_sg = SceneGraph(
        [
            SceneNode(
                transform=Tform3(Vec3(0, 0, 0)),
                shapes=[
                    Triangle(
                        to_raw_pnt(VAR_VAL, Pnt3(Const(-8), Const(-8), Const(1.0) * VAR_T)),
                        to_raw_pnt(VAR_VAL, Pnt3(Const(8) * VAR_T, Const(-8), Const(0 - 1e-4))),
                        to_raw_pnt(VAR_VAL, Pnt3(Const(-8), Const(8)  * VAR_T, Const(-1e-3))),
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

    aspect_ratio = image_width / image_height
    light_cam = TurntableCamera(dist=10, angle=0, aspect_ratio=aspect_ratio, fov_angle_y=90*np.pi/180)
    rt = RayTraceThresholdLambertianPotto(light_sg, light_cam, 10)

    start = time.time()
    potto_img = rt.ray_trace_parallel(image_width, image_height, is_primal)
    end = time.time()
    mode = "primal" if is_primal else "deriv"
    print(f"rendering {mode} image of {image_width}x{image_height} took {end - start} seconds")

    plt.axis('off')
    plt.imshow(potto_img)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True, choices=["small", "medium", "large"], help="The size of rendered image: Small: 4x4, Medium: 16x16, Large: 40x40")
    parser.add_argument("--mode", required=True, choices=["primal", "gradient"], help="TODO:")
    args = parser.parse_args()
    image_width, image_height = 0, 0
    if args.size == "small":
        image_width, image_height = 4, 4
    elif args.size == "medium":
        image_width, image_height = 16, 16
    else:
        image_width, image_height = 40, 40
    is_primal = args.mode == "primal"
    main(image_width, image_height, is_primal)
