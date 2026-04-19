"""Blender helpers for loading and rendering Holodeck scene JSONs.

Limitations:
- Wall/floor materials are approximated from the JSON's `wall_design` / `floor_design`
  text hints; Holodeck's THOR material names don't map to textures we have on disk.
- Door/window assets are rendered as proxy boxes sized from
  `~/.objathor-assets/holodeck/<version>/{doors,windows}/{door,window}-database.json`,
  not as the actual Unity prefabs.
- Unity light intensity -> Blender Watts is heuristic.

Coordinate convention: Unity is left-handed Y-up, Blender is right-handed Z-up.
We map (ux, uy, uz) -> (ux, uz, uy). Unity Y-axis rotation (yaw) becomes Blender
Z-axis rotation with sign flipped to preserve handedness.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable, Sequence

# isort: off

import bpy  # type: ignore
import bmesh  # type: ignore
from mathutils import Euler, Vector  # type: ignore
from mathutils.geometry import tessellate_polygon  # type: ignore

# isort: on

from ai2holodeck.constants import (
    HOLODECK_BASE_DATA_DIR,
    OBJATHOR_ASSETS_DIR,
)

DOOR_DB_PATH = os.path.join(
    HOLODECK_BASE_DATA_DIR, "doors", "door-database.json"
)
WINDOW_DB_PATH = os.path.join(
    HOLODECK_BASE_DATA_DIR, "windows", "window-database.json"
)
MATERIAL_IMAGES_DIR = os.path.join(
    HOLODECK_BASE_DATA_DIR, "materials", "images"
)

# Unity-intensity -> Blender heuristic multipliers.
SUN_WATTS_PER_UNIT = 3.0
POINT_WATTS_PER_UNIT = 600.0

# Default wall thickness so walls are visible from above (pitch ~90).
WALL_THICKNESS = 0.08


# ---------- coord helpers ----------


def u2b(p: dict | Sequence[float]) -> Vector:
    """Unity (x, y, z) dict or seq -> Blender Vector(x, z, y)."""
    if isinstance(p, dict):
        x, y, z = p["x"], p.get("y", 0.0), p["z"]
    else:
        x, y, z = (
            p[0],
            p[1] if len(p) > 1 else 0.0,
            p[2] if len(p) > 2 else 0.0,
        )
    return Vector((x, z, y))


def u2b_xz(p: dict | Sequence[float]) -> tuple[float, float]:
    """Unity floor-plane point -> Blender (x, y) ignoring height."""
    if isinstance(p, dict):
        return p["x"], p["z"]
    return p[0], p[1] if len(p) > 1 else 0.0


# ---------- scene reset ----------


def clear() -> None:
    """Wipe all data blocks for a fresh scene."""
    for c in bpy.data.collections:
        for obj in list(c.objects):
            c.objects.unlink(obj)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.node_groups,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.worlds,
    ):
        for block in list(coll):
            coll.remove(block, do_unlink=True)


# ---------- materials ----------

# Keyword -> (base color RGB 0-1, roughness)
_DESIGN_PALETTE: list[
    tuple[tuple[str, ...], tuple[float, float, float], float]
] = [
    (("oak", "wood", "hardwood", "timber", "plank"), (0.42, 0.27, 0.15), 0.55),
    (("marble",), (0.92, 0.91, 0.88), 0.25),
    (("tile", "tiled"), (0.85, 0.85, 0.83), 0.35),
    (("carpet", "rug"), (0.45, 0.40, 0.36), 0.95),
    (("concrete", "cement"), (0.62, 0.62, 0.60), 0.85),
    (("brick",), (0.55, 0.30, 0.24), 0.90),
    (("beige", "cream", "ivory", "tan"), (0.86, 0.78, 0.66), 0.70),
    (("white", "drywall"), (0.92, 0.92, 0.90), 0.75),
    (("grey", "gray", "slate"), (0.55, 0.55, 0.55), 0.75),
    (("black", "dark"), (0.14, 0.14, 0.14), 0.60),
    (("blue",), (0.30, 0.45, 0.75), 0.70),
    (("green",), (0.35, 0.55, 0.40), 0.70),
    (("red", "crimson"), (0.65, 0.20, 0.20), 0.70),
    (("yellow", "gold"), (0.85, 0.70, 0.30), 0.55),
]


def _design_to_rgba(
    text: str | None, fallback: tuple[float, float, float] = (0.75, 0.75, 0.75)
) -> tuple[tuple[float, float, float], float]:
    if not text:
        return fallback, 0.7
    t = text.lower()
    for keys, rgb, rough in _DESIGN_PALETTE:
        if any(k in t for k in keys):
            return rgb, rough
    return fallback, 0.7


def _make_principled_material(
    name: str,
    rgb: tuple[float, float, float],
    roughness: float,
    alpha: float = 1.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if alpha < 1.0:
            mat.blend_method = (
                "BLEND" if hasattr(mat, "blend_method") else "BLEND"
            )
    return mat


def _resolve_material_image(material_name: str | None) -> str | None:
    if not material_name or not os.path.isdir(MATERIAL_IMAGES_DIR):
        return None
    candidates = [material_name, material_name.removeprefix("Wall")]
    for cand in candidates:
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(MATERIAL_IMAGES_DIR, f"{cand}{ext}")
            if os.path.isfile(p):
                return p
    # Case-insensitive fallback (in case filesystem is case-sensitive).
    try:
        listing = {f.lower(): f for f in os.listdir(MATERIAL_IMAGES_DIR)}
    except OSError:
        return None
    for cand in candidates:
        for ext in (".png", ".jpg", ".jpeg"):
            real = listing.get(f"{cand}{ext}".lower())
            if real:
                return os.path.join(MATERIAL_IMAGES_DIR, real)
    return None


def _make_textured_material(
    name: str, image_path: str, roughness: float = 0.7, uv_scale: float = 1.0
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        return _make_principled_material(name, (0.75, 0.75, 0.75), roughness)
    tex = nt.nodes.new("ShaderNodeTexImage")
    try:
        tex.image = bpy.data.images.load(image_path, check_existing=True)
    except RuntimeError:
        return _make_principled_material(name, (0.75, 0.75, 0.75), roughness)
    if uv_scale != 1.0:
        mapping = nt.nodes.new("ShaderNodeMapping")
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping.inputs["Scale"].default_value = (uv_scale, uv_scale, 1.0)
        nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def _surface_material(
    prefix: str,
    material_field: dict | None,
    design_text: str | None,
    roughness_default: float,
    uv_scale: float = 1.0,
) -> bpy.types.Material:
    """Try image lookup from material.name, else fall back to design-text palette."""
    name = (material_field or {}).get("name") if material_field else None
    img = _resolve_material_image(name)
    if img is not None:
        return _make_textured_material(
            f"{prefix}_{name}",
            img,
            roughness=roughness_default,
            uv_scale=uv_scale,
        )
    rgb, rough = _design_to_rgba(design_text)
    return _make_principled_material(f"{prefix}_fallback", rgb, rough)


# ---------- room shell ----------


def _wall_basis(wall: dict) -> tuple[Vector, Vector, Vector, float, float]:
    """Return (origin_blender, along_unit_blender, up_unit_blender, length, height).

    `origin` is the bottom-left wall corner (smallest along-axis param). The wall
    polygon has 4 points: bottom-A, top-A, top-B, bottom-B (Unity coords).
    """
    poly = wall["polygon"]
    p0 = u2b(poly[0])
    p1 = u2b(poly[1])  # directly above p0
    p3 = u2b(poly[3])  # along the wall from p0
    height = (p1 - p0).length
    length = (p3 - p0).length
    along = (
        (p3 - p0).normalized() if length > 1e-6 else Vector((1.0, 0.0, 0.0))
    )
    up = Vector((0.0, 0.0, 1.0))
    return p0, along, up, length, height


def _holes_for_wall(
    wall_id: str, openings: Iterable[dict]
) -> list[tuple[float, float, float, float]]:
    """List of (x_min, y_min, x_max, y_max) hole rects in wall-local coords."""
    holes: list[tuple[float, float, float, float]] = []
    for o in openings:
        if o.get("wall0") != wall_id and o.get("wall1") != wall_id:
            continue
        hp = o["holePolygon"]
        x0, y0 = hp[0]["x"], hp[0]["y"]
        x1, y1 = hp[1]["x"], hp[1]["y"]
        holes.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    return holes


def _build_wall_with_holes(
    wall: dict, openings: Iterable[dict], material: bpy.types.Material
) -> bpy.types.Object | None:
    origin, along, up, length, height = _wall_basis(wall)
    if length <= 1e-4 or height <= 1e-4:
        return None

    outer = [(0.0, 0.0), (length, 0.0), (length, height), (0.0, height)]
    holes_2d = _holes_for_wall(wall["id"], openings)
    # Clip holes to wall bounds.
    clipped_holes: list[list[tuple[float, float]]] = []
    for x0, y0, x1, y1 in holes_2d:
        x0c, x1c = max(0.0, x0), min(length, x1)
        y0c, y1c = max(0.0, y0), min(height, y1)
        if x1c - x0c < 1e-3 or y1c - y0c < 1e-3:
            continue
        clipped_holes.append([(x0c, y0c), (x1c, y0c), (x1c, y1c), (x0c, y1c)])

    # Build vertex list: outer first, then each hole.
    verts_2d: list[tuple[float, float]] = list(outer) + [
        v for h in clipped_holes for v in h
    ]
    verts_3d = [(x, y, 0.0) for x, y in verts_2d]
    loops = [
        [Vector((x, y, 0.0)) for x, y in outer],
        *[[Vector((x, y, 0.0)) for x, y in h] for h in clipped_holes],
    ]
    tris = tessellate_polygon(loops)

    mesh = bpy.data.meshes.new(f"mesh_{wall['id']}")
    mesh.from_pydata(verts_3d, [], [list(t) for t in tris])
    mesh.update()
    # Wall-local UV: u = along (0..length), v = up (0..height). 1m per tile.
    _set_planar_uvs(mesh, verts_2d)
    mesh.update()

    obj = bpy.data.objects.new(wall["id"], mesh)
    bpy.context.scene.collection.objects.link(obj)

    # Place: vertex (u, v, 0) -> origin + u*along + v*up.
    # Build a 4x4 matrix manually.
    from mathutils import Matrix

    n = along.cross(up)
    m = Matrix(
        (
            (along.x, up.x, n.x, origin.x),
            (along.y, up.y, n.y, origin.y),
            (along.z, up.z, n.z, origin.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    obj.matrix_world = m
    obj.data.materials.append(material)
    return obj


def _polygon_to_mesh(
    name: str,
    polygon_xz: list[tuple[float, float]],
    y: float,
    flip: bool = False,
) -> bpy.types.Object:
    """Triangulate a 2D floor/ceiling polygon at given height (Blender Z=y)."""
    loop = [Vector((x, z, 0.0)) for x, z in polygon_xz]
    tris = tessellate_polygon([loop])
    if flip:
        tris = [(t[0], t[2], t[1]) for t in tris]
    verts = [(x, z, y) for x, z in polygon_xz]
    mesh = bpy.data.meshes.new(f"mesh_{name}")
    mesh.from_pydata(verts, [], [list(t) for t in tris])
    mesh.update()
    # Planar UVs: 1 unit = 1 meter; texture loaded at uv_scale handles tiling.
    _set_planar_uvs(mesh, [(x, z) for x, z in polygon_xz])
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _set_planar_uvs(
    mesh: bpy.types.Mesh, uv_per_vertex: list[tuple[float, float]]
) -> None:
    if mesh.uv_layers:
        uvl = mesh.uv_layers[0]
    else:
        uvl = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            uvl.data[li].uv = uv_per_vertex[vi]


def _solidify(obj: bpy.types.Object, thickness: float) -> None:
    """Add small two-sided thickness to a flat mesh."""
    mod = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 0.0  # symmetric around the original surface


def build_shell(scene: dict, hide_ceiling: bool = True) -> None:
    wall_height = float(scene.get("wall_height", 2.7))
    rooms = scene.get("rooms", [])
    walls = scene.get("walls", [])
    doors = scene.get("doors", []) or []
    windows = scene.get("windows", []) or []
    openings = list(doors) + list(windows)

    # Floor + ceiling per room.
    for room in rooms:
        poly_xz = [(p["x"], p["z"]) for p in room["floorPolygon"]]
        floor_mat = _surface_material(
            f"floor_{room['id']}",
            room.get("floorMaterial"),
            room.get("floor_design"),
            roughness_default=0.55,
            uv_scale=0.5,
        )
        floor = _polygon_to_mesh(f"floor_{room['id']}", poly_xz, y=0.0)
        floor.data.materials.append(floor_mat)

        if hide_ceiling:
            continue
        proc = scene.get("proceduralParameters", {}) or {}
        ceil_mat = _surface_material(
            f"ceil_{room['id']}",
            proc.get("ceilingMaterial"),
            None,
            roughness_default=0.85,
        )
        ceil = _polygon_to_mesh(
            f"ceiling_{room['id']}", poly_xz, y=wall_height, flip=True
        )
        ceil.data.materials.append(ceil_mat)

    # Walls.
    wall_design_by_room = {r["id"]: r.get("wall_design") for r in rooms}
    for wall in walls:
        mat = _surface_material(
            f"mat_{wall['id']}",
            wall.get("material"),
            wall_design_by_room.get(wall.get("roomId")),
            roughness_default=0.75,
        )
        obj = _build_wall_with_holes(wall, openings, mat)
        if obj is not None:
            obj["holodeck_wall"] = True
            _solidify(obj, WALL_THICKNESS)


def cull_near_walls(scene: dict, cam_obj: bpy.types.Object) -> None:
    """Hide walls between camera and scene center so the interior is visible."""
    center, _ = _scene_bbox(scene)
    cam_xy = Vector((cam_obj.location.x, cam_obj.location.y))
    center_xy = Vector((center.x, center.y))
    view_dir = center_xy - cam_xy
    if view_dir.length < 1e-6:
        return
    view_dir.normalize()
    for obj in bpy.data.objects:
        if not obj.get("holodeck_wall"):
            continue
        # Wall world center on the floor plane.
        wc = obj.matrix_world @ Vector((0.0, 0.0, 0.0))
        # Approx: average with the far corner using local bbox.
        if obj.data and obj.data.vertices:
            local_center = sum(
                (Vector(v.co) for v in obj.data.vertices), Vector()
            ) / len(obj.data.vertices)
            wc = obj.matrix_world @ local_center
        wc_xy = Vector((wc.x, wc.y))
        # Hide if wall center is on the camera side of the scene center.
        on_camera_side = (wc_xy - center_xy).dot(cam_xy - center_xy) > 0
        # Also require it's roughly between cam and center along view dir.
        proj = (wc_xy - cam_xy).dot(view_dir)
        between = 0 < proj < (center_xy - cam_xy).length
        hide = on_camera_side and between
        obj.hide_render = hide
        obj.hide_viewport = hide


# ---------- objects ----------


def _glb_path(asset_id: str) -> str | None:
    p = os.path.join(OBJATHOR_ASSETS_DIR, asset_id, f"{asset_id}.glb")
    return p if os.path.isfile(p) else None


def _selected_top_level_objects(before: set[str]) -> list[bpy.types.Object]:
    return [o for o in bpy.context.scene.objects if o.name not in before]


def place_objects(scene: dict) -> None:
    objects = scene.get("objects", []) or []
    n_loaded = n_skipped = 0
    for entry in objects:
        asset_id = entry["assetId"]
        path = _glb_path(asset_id)
        if path is None:
            print(f"[render] skip missing GLB: {asset_id}")
            n_skipped += 1
            continue
        before = {o.name for o in bpy.context.scene.objects}
        bpy.ops.import_scene.gltf(filepath=path)
        new_objs = _selected_top_level_objects(before)
        all_new = [
            o for o in bpy.context.scene.objects if o.name not in before
        ]
        if not new_objs:
            n_skipped += 1
            continue

        # Holodeck/Unity convention: position is the bbox CENTER. Objathor GLBs
        # are typically authored with origin at the bbox bottom-center, so we
        # shift the imported geometry so its bbox center sits at the local origin.
        bbox_min, bbox_max = _world_bbox(all_new)
        bbox_center = (bbox_min + bbox_max) * 0.5
        for o in new_objs:
            if o.parent is None:
                o.location -= bbox_center

        # Group under an empty so we can transform as a unit.
        empty = bpy.data.objects.new(f"obj_{entry.get('id', asset_id)}", None)
        bpy.context.scene.collection.objects.link(empty)
        for o in new_objs:
            if o.parent is None:
                o.parent = empty

        pos = u2b(entry["position"])
        empty.location = pos
        rot = entry.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
        # Unity Y-yaw is left-handed; flip sign for right-handed Blender Z.
        empty.rotation_euler = Euler(
            (
                math.radians(rot.get("x", 0.0)),
                math.radians(rot.get("z", 0.0)),
                -math.radians(rot.get("y", 0.0) + 180.0),
            ),
            "XYZ",
        )
        n_loaded += 1
    print(f"[render] loaded {n_loaded} objects, skipped {n_skipped}")


def _world_bbox(objs: Iterable[bpy.types.Object]) -> tuple[Vector, Vector]:
    big = float("inf")
    mn = Vector((big, big, big))
    mx = Vector((-big, -big, -big))
    for o in objs:
        if o.type != "MESH" or o.data is None:
            continue
        mw = o.matrix_world
        for v in o.data.vertices:
            wv = mw @ Vector(v.co)
            mn.x, mn.y, mn.z = (
                min(mn.x, wv.x),
                min(mn.y, wv.y),
                min(mn.z, wv.z),
            )
            mx.x, mx.y, mx.z = (
                max(mx.x, wv.x),
                max(mx.y, wv.y),
                max(mx.z, wv.z),
            )
    return mn, mx


# ---------- doors / windows ----------

_door_db_cache: dict | None = None
_window_db_cache: dict | None = None


def _load_db(path: str, cache_attr: str) -> dict:
    g = globals()
    if g[cache_attr] is None:
        with open(path) as f:
            g[cache_attr] = json.load(f)
    return g[cache_attr]


def _opening_proxy(
    opening: dict, db: dict, frame_mat: bpy.types.Material, depth: float = 0.08
) -> None:
    asset_id = opening["assetId"]
    entry = db.get(asset_id)
    if entry is None:
        return
    bb = entry.get("boundingBox") or entry.get("size")
    if bb is None:
        return
    sx, sy, sz = bb["x"], bb["y"], bb["z"]
    pos = u2b(opening["assetPosition"])

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(pos.x, pos.y, pos.z))
    obj = bpy.context.active_object
    obj.name = opening["id"]
    # Unity y is height -> Blender z; depth is min dimension.
    obj.scale = (max(sx, depth), depth, max(sy, depth))
    obj.data.materials.append(frame_mat)


def place_openings(scene: dict) -> None:
    door_mat = _make_principled_material("door_proxy", (0.35, 0.22, 0.14), 0.6)
    window_mat = _make_principled_material(
        "window_proxy", (0.55, 0.75, 0.85), 0.05, alpha=0.3
    )
    if os.path.isfile(DOOR_DB_PATH):
        db = _load_db(DOOR_DB_PATH, "_door_db_cache")
        for d in scene.get("doors", []) or []:
            _opening_proxy(d, db, door_mat)
    if os.path.isfile(WINDOW_DB_PATH):
        db = _load_db(WINDOW_DB_PATH, "_window_db_cache")
        for w in scene.get("windows", []) or []:
            _opening_proxy(w, db, window_mat)


# ---------- lights ----------


def add_lights(scene: dict) -> None:
    for i, light in enumerate(
        scene.get("proceduralParameters", {}).get("lights", [])
    ):
        ltype = light.get("type", "point").lower()
        intensity = float(light.get("intensity", 1.0))
        rgb = light.get("rgb", {"r": 1.0, "g": 1.0, "b": 1.0})
        color = (rgb["r"], rgb["g"], rgb["b"])
        if ltype == "directional":
            ld = bpy.data.lights.new(name=f"sun_{i}", type="SUN")
            ld.energy = intensity * SUN_WATTS_PER_UNIT
            ld.color = color
            obj = bpy.data.objects.new(f"sun_{i}", ld)
            bpy.context.scene.collection.objects.link(obj)
            rot = light.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
            obj.rotation_euler = Euler(
                (
                    math.radians(rot.get("x", 0.0)),
                    math.radians(rot.get("z", 0.0)),
                    -math.radians(rot.get("y", 0.0)),
                ),
                "XYZ",
            )
        elif ltype == "point":
            ld = bpy.data.lights.new(name=f"point_{i}", type="POINT")
            ld.energy = intensity * POINT_WATTS_PER_UNIT
            ld.color = color
            obj = bpy.data.objects.new(f"point_{i}", ld)
            bpy.context.scene.collection.objects.link(obj)
            obj.location = u2b(light["position"])
        else:
            continue


# ---------- world (background + HDRI) ----------


def set_world(
    bg_color_rgb: tuple[float, float, float], hdri_path: str | None
) -> None:
    world = bpy.data.worlds.new("HolodeckWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputWorld")
    out.location = (600, 0)

    bg_color = nt.nodes.new("ShaderNodeBackground")
    bg_color.inputs["Color"].default_value = (*bg_color_rgb, 1.0)
    bg_color.inputs["Strength"].default_value = 1.0
    bg_color.location = (0, -150)

    if hdri_path and os.path.isfile(hdri_path):
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path)
        env.location = (-300, 150)
        bg_env = nt.nodes.new("ShaderNodeBackground")
        bg_env.inputs["Strength"].default_value = 1.0
        bg_env.location = (0, 150)
        nt.links.new(env.outputs["Color"], bg_env.inputs["Color"])

        # Mix: camera ray sees solid color, lighting samples see HDRI.
        lp = nt.nodes.new("ShaderNodeLightPath")
        lp.location = (0, 350)
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.location = (300, 0)
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg_env.outputs["Background"], mix.inputs[1])
        nt.links.new(bg_color.outputs["Background"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    else:
        nt.links.new(bg_color.outputs["Background"], out.inputs["Surface"])


# ---------- camera ----------


def _scene_bbox(scene: dict) -> tuple[Vector, float]:
    xs: list[float] = []
    zs: list[float] = []
    for room in scene.get("rooms", []):
        for p in room.get("floorPolygon", []):
            xs.append(p["x"])
            zs.append(p["z"])
    if not xs:
        return Vector((0.0, 0.0, 1.0)), 5.0
    cx = (min(xs) + max(xs)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    height = float(scene.get("wall_height", 2.7))
    diag = math.hypot(max(xs) - min(xs), max(zs) - min(zs))
    return Vector((cx, cz, height / 2.0)), diag


def orbit_camera(
    scene: dict,
    pitch_deg: float,
    yaw_deg: float,
    focal_mm: float,
    fit_ratio: float = 0.0,
    aspect_ratio: float = 1.0,
    cull_walls: bool = True,
) -> bpy.types.Object:
    """Place an orbit camera aimed at scene center.

    `fit_ratio` (0..1) matches genxr's `render_perspective` semantics:
      - 0.0: bounding-sphere distance (safe, looser framing).
      - 1.0: tight-fit distance (every visible vertex exactly inside frustum).
      - between: linear interpolation.

    If `cull_walls`, walls between camera and scene center are hidden before
    visible-vertex collection — so fit_ratio frames what will actually be
    rendered.
    """
    center, diag = _scene_bbox(scene)

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = focal_mm
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.01
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    # Direction from scene center toward the camera (unit).
    back = Vector(
        (
            math.cos(pitch) * math.sin(yaw),
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
        )
    )

    # Provisional camera placement so wall culling can use its location.
    provisional_r = max(diag, 4.0) * 2.0
    cam_obj.location = center + back * provisional_r
    if cull_walls:
        cull_near_walls(scene, cam_obj)

    verts = _visible_world_vertices()
    if verts:
        radius = max((v - center).length for v in verts)
    else:
        radius = max(diag * 0.6, 2.0)

    fov_x = cam_data.angle
    fit_ratio = max(0.0, min(1.0, fit_ratio))
    sphere_dist = radius / math.sin(fov_x / 2)

    if fit_ratio > 0.0 and verts:
        fov_y = 2 * math.atan(math.tan(fov_x / 2) / aspect_ratio)
        forward = -back
        up_hint = Vector((0.0, 0.0, 1.0))
        right = forward.cross(up_hint)
        if right.length < 1e-6:
            right = Vector((1.0, 0.0, 0.0))
        right.normalize()
        up = right.cross(forward).normalized()
        fit_dist = 0.0
        for v in verts:
            r = v - center
            vx = r.dot(right)
            vy = r.dot(up)
            vz = r.dot(-forward)
            dist_x = vz + abs(vx) / math.tan(fov_x / 2)
            dist_y = vz + abs(vy) / math.tan(fov_y / 2)
            fit_dist = max(fit_dist, dist_x, dist_y)
        distance = sphere_dist * (1 - fit_ratio) + fit_dist * fit_ratio
    else:
        distance = sphere_dist

    cam_obj.location = center + back * distance
    cam_data.clip_end = distance + 2 * radius * 1.01

    direction = (center - cam_obj.location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.context.scene.camera = cam_obj
    return cam_obj


def _visible_world_vertices() -> list[Vector]:
    verts: list[Vector] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render or obj.hide_viewport:
            continue
        if obj.data is None:
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            verts.append(mw @ Vector(v.co))
    return verts


# ---------- render ----------


def render(filepath: str, resolution: int, samples: int, engine: str, transparent: bool = False) -> None:
    sc = bpy.context.scene
    sc.render.engine = engine
    sc.render.resolution_x = resolution
    sc.render.resolution_y = resolution
    sc.render.resolution_percentage = 100
    sc.render.filepath = filepath
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    sc.render.film_transparent = transparent
    if engine == "CYCLES":
        sc.cycles.samples = samples
        sc.cycles.device = "CPU"
    elif engine.startswith("BLENDER_EEVEE"):
        try:
            sc.eevee.taa_render_samples = samples
        except AttributeError:
            pass
    result = bpy.ops.render.render(write_still=True)
    if "FINISHED" not in result:
        raise RuntimeError(f"render failed: {result}")
    if not os.path.isfile(filepath):
        raise RuntimeError(
            f"render returned FINISHED but no file at {filepath}"
        )
