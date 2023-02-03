from __future__ import annotations
from dataclasses import dataclass
from abc import abstractmethod, ABC
from enum import Enum
import math

from graphics.util.ray3 import Ray3, Hit3
from graphics.util.mat3 import Tform3, Mat3
from graphics.util.vec3 import Pnt3, Vec3
from graphics.util.vec2 import Pnt2, Vec2
from graphics.util.util import remap


@dataclass()
class Camera(ABC):
    """
    transform: idk what this is
    frustrum_dist_from_cam:
    """

    transform: Tform3 = Tform3()
    focus_distance: float = 25
    fov_angle_y: float = 90 * math.pi/180
    aspect_ratio: float = 4/3

    def up(self):
        """
        normalized local up direction
        """
        return Vec3(0, 1, 0)

    def forward(self):
        """
        normalized local forward direction
        """
        return Vec3(0, 0, -1)

    @classmethod
    def look_at(cls, eye: Pnt3, look_point: Pnt3, up: Vec3) -> Tform3:
        """
        Creates a transform for a camera that puts the camera at :eye: and makes it look at :look_point:,
          with the camera's up oriented to world :up:

        By convention, a camera looks forward along the negative-z axis
        """
        # See https://math.stackexchange.com/a/476311/415088
        # Rotate the camera look point to the normed look point
        tlp: Vec3 = look_point - eye
        normed_look = tlp / tlp.norm()
        camera_look = Vec3(0, 0, -1)

        if normed_look == camera_look:
            mat = Mat3()
        else:
            # v is the normal to the plane containting both vectors
            v = camera_look.cross(normed_look)
            cos = camera_look.dot(normed_look)
            sin = v.norm()

            v_cross = Mat3(
                Vec3(0, -v[2], v[1]), Vec3(v[2], 0, -v[0]), Vec3(-v[1], v[0], 0)
            )
            mat = Mat3() + v_cross + v_cross @ v_cross * (1 - cos) / sin**2

        return Tform3(translate=eye - look_point, rot_scale=mat)

    def screen_point_to_ndc(self, pnt: Pnt2, screen_width, screen_height) -> Pnt2:
        """Put a point in screen coordinates into normalized device coordinates.

        pnt: (x, y) where (0, 0) is the lower left corner of the screen.

        Returns the ndc of the input
        # TODO: refactor into image?
        """
        ndc_x = remap(pnt.x, (0, screen_width), (-1, 1))
        ndc_y = remap(pnt.y, (0, screen_height), (-1, 1))
        return Pnt2(ndc_x, ndc_y)

    def ndc_to_ray(self, pnt: Pnt2) -> Ray3:
        """Put a point in normalized device coordinates into world coordinates.

        pnt: (x, y, 0) with (x, y) in the range [-1, 1] x [-1, 1]

        Returns a normalized ray in the direction of pnt from the camera.
        """
        # Form the frustrum in world space, which corresponds to the origin in centered screen space
        cam_loc = Pnt3()
        look_dir = self.forward()
        frustrum_center: Pnt3 = cam_loc + self.focus_distance * look_dir

        # Set up basis on the plane
        up_on_plane = self.up()
        right_on_plane = look_dir.cross(up_on_plane)

        # Build the ray to the camera
        half_fov_y = self.focus_distance * math.tan(self.fov_angle_y / 2)
        horiz_shift: Vec3 = pnt.x * right_on_plane * half_fov_y * self.aspect_ratio
        vert_shift: Vec3 = pnt.y * up_on_plane * half_fov_y

        # Point on the screen in world space
        to_point: Pnt3 = horiz_shift + vert_shift + frustrum_center
        dir = to_point - cam_loc
        dir: Vec3 = dir / dir.norm()
        return self.transform @ Ray3(cam_loc, dir)

    def diff_ndc_to_ray(
        self, pnt: Pnt3
    ) -> Ray3:
        """Put a point in screen coordinates into world coordinates.

        pnt: (x, y, 0) where (0, 0) is the lower left corner of the frustrum.
        screen_width: the width of frustrum used for displaying the image
        screen_height: the height of frustrum used for displaying the image

        Returns a normal ray in the direction of pnt from the camera.
        """
        look_dir = self.transform @ Vec3(0, 0, -1)

        # Form the frustrum in world space, which corresponds to the origin in centered screen space
        cam_loc = Pnt3()
        look_dir = self.forward()
        frustrum_center: Pnt3 = cam_loc + self.focus_distance * look_dir

        # Set up basis on the plane
        up_on_plane = self.up()
        right_on_plane = look_dir.cross(up_on_plane)

        # Build the ray to the camera
        half_fov_y = self.focus_distance * math.tan(self.fov_angle_y / 2)
        horiz_shiftx = pnt.x * right_on_plane.x * half_fov_y * self.aspect_ratio
        horiz_shifty = pnt.x * right_on_plane.y * half_fov_y * self.aspect_ratio
        horiz_shiftz = pnt.x * right_on_plane.z * half_fov_y * self.aspect_ratio
        horiz_shift = Vec3(horiz_shiftx, horiz_shifty, horiz_shiftz)

        vert_shiftx = pnt.y * up_on_plane.x * half_fov_y
        vert_shifty = pnt.y * up_on_plane.y * half_fov_y
        vert_shiftz = pnt.y * up_on_plane.z * half_fov_y
        vert_shift = Vec3(vert_shiftx, vert_shifty, vert_shiftz)

        # Point on the screen in world space
        to_point: Pnt3 = horiz_shift + vert_shift + frustrum_center
        dir = to_point - cam_loc

        # remove the norm for now
        # n = dir.potto_norm()
        # dirx = dir.x / n
        # diry = dir.y / n
        # dirz = dir.z / n
        # dir = Vec3(dirx, diry, dirz)
        return self.transform @ Ray3(cam_loc, dir)


@dataclass()
class TurntableCamera(Camera):
    """
    A 3d turntable camera
    It looks at the origin
    """

    dist: float = 50  # distance from the origin
    angle: float = 0  # (degrees) the camera starts at (0, 0, dist) when angle==0, and goes CCW around the y axis

    @property
    def transform(self):
        cam_radians = self.angle * math.pi / 180
        eye = self.dist * Pnt3(math.sin(cam_radians), 0, math.cos(cam_radians))
        origin = Pnt3()
        up = Vec3(0, 1, 0)
        return Camera.look_at(eye, origin, up)

    @transform.setter
    def transform(self, t):
        """Do nothing. Implemented to prevent Attribute Error"""  # HACK?
