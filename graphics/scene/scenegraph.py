from __future__ import annotations
from typing import TypeVar, List, Tuple
from dataclasses import dataclass, field
from collections.abc import Iterable
from itertools import chain

from graphics.util.mat3 import Tform3
from graphics.util.ray3 import Ray3, Hit3
from graphics.util.dpg_util import to_mvmat4
from graphics.scene.shape3 import Shape3, RenderMode
from graphics.scene.camera import Camera
from graphics.scene.light import Light

from potto import (
    TegVar,
)


deltax = TegVar("deltax")
deltay = TegVar("deltay")


@dataclass
class SceneGraph(object):
    """
    A parent-child tree of scene objects
    """

    nodes: List[SceneNode] = field(default_factory=list)
    lights: List[Light] = field(default_factory=list)

    def render(self, dpg):
        for n in self.nodes:
            n.render(dpg)

    def pick(self, r: Ray3) -> Iterable[Tuple[SceneNode, Hit3]]:  # unaccelerated brute force search, returns all hits
        return chain(*(n.pick(r) for n in self.nodes))


@dataclass
class SceneNode(object):
    """
    A scene object, which is a bunch of shapes under a transformation relative to its parent
    """

    transform: Tform3 = Tform3()  # local space in terms of parent space
    shapes: List[Shape3] = field(default_factory=list)
    children: List[SceneNode] = field(default_factory=list)
    parent: SceneNode | None = None  # TODO: have helper functions that construct this when adding children?

    def world_transform(self):  # local in terms of world
        return self.parent_transform() @ self.transform

    def parent_transform(self):
        match self.parent:
            case None:
                return Tform3()
            case SceneNode() as p:
                return p.parent_transform @ p.transform

    def render(self, dpg, mode: RenderMode = RenderMode.BASIC):
        with dpg.draw_node() as n_id:
            dpg.apply_transform(n_id, to_mvmat4(self.transform))
            for s in self.shapes:
                s.render(dpg)
            for c in self.children:
                c.render(dpg)

    def pick(self, r: Ray3) -> Iterable[Tuple[SceneNode, Hit3]]:
        local_ray = self.transform.inverse() @ r
        isects = (s.intersect(local_ray) for s in self.shapes)
        hits = ((self, i) for i in isects if i.t is not None)
        return chain(hits, *(c.pick(r) for c in self.children))


@dataclass
class Viewport(object):
    """
    A 3d viewport that can views a scenegraph through a camera
    """

    scene: SceneGraph
    camera: Camera
    selection: List[SceneNode] = field(default_factory=list)

    def render(self, dpg):  # viewport rendering for selection, not a full-blown raytracer
        pass  # TODO abstract code from main_workspace into here

    def select(self, r: Ray3, add=False):
        if not add:
            self.selection = []
        all_picked = self.scene.pick(self.camera.transform @ r)
        if all_picked:
            node, _ = min(all_picked, key=lambda n, h: h)
            self.selection.append(node)
