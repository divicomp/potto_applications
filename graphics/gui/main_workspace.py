import dearpygui.dearpygui as dpg
import math
from graphics.scene.scenegraph import Viewport, SceneGraph, SceneNode, RayTrace, VAR_T
from graphics.scene.shape3 import Triangle
from graphics.scene.camera import TurntableCamera
from graphics.gui.image import PixGrid
from graphics.util.color import Colors
from graphics.util.vec3 import Pnt3, Vec3
from graphics.util.mat3 import Tform3

from potto import Const


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
                        Pnt3(Const(-20) + Const(.2), Const(-20) + Const(.5), Const(0.1)),
                        Pnt3(Const(20) * VAR_T + Const(.5), Const(-20) + Const(-.6), Const(0.02)),
                        Pnt3(Const(-20) + Const(-.3), Const(20) * VAR_T + Const(-.1), Const(-0.2)),
                        e=Colors.WHITE,
                    ),
                ],
            )
        ]
    )
    cam = TurntableCamera(dist=25, angle=0)
    vp = Viewport(sg, cam)
    rt = RayTrace(sg, cam, 10)
    raypix = PixGrid(width=1, height=1)

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
                    proj_mat = dpg.create_perspective_matrix(
                        fov=fov * math.pi / 180.0, aspect=1.0, zNear=0.1, zFar=100
                    )
                    dpg.apply_transform(proj_id, proj_mat)

                    with dpg.draw_node(tag="turntable camera"):
                        set_turntable_angle(None, 0)

                        # axes arrows
                        dpg.draw_arrow(
                            (4, 0, 0), (0, 0, 0), color=(255, 0, 0, 150), size=1
                        )
                        dpg.draw_arrow(
                            (0, 4, 0), (0, 0, 0), color=(0, 255, 0, 150), size=1
                        )
                        dpg.draw_arrow(
                            (0, 0, 4), (0, 0, 0), color=(0, 0, 255, 150), size=1
                        )

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
        raypix.img = rt.diff_ray_trace(raypix.width, raypix.height)
        # raypix.img = rt.diff_ray_trace_parallel(raypix.width, raypix.height)
        raypix.update_raw_array()
        print("do_render has finished")

    with dpg.window(label="raytracer", width=550, height=550) as raytrace_id:
        dpg.set_item_pos(raytrace_id, [600, 0])
        dpg.add_image(
            raypix.gui_id, width=500, height=500
        )  # TODO fix the default antialiasing

    with dpg.window(label="raytrace params") as camera_param_id:
        button_id = dpg.add_button(label="do render", callback=do_render)
        dpg.set_item_pos(camera_param_id, [630, 580])

    do_render()
    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
