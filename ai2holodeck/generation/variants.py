"""Post-generation scene variants.

All variants operate on an already-generated ``scene`` dict. They mutate
``scene["objects"]`` (the placed floor + wall + small objects) **only** and
never re-run the LLM, the DFS placement solver, or the AI2-THOR / Unity
controller. This keeps them essentially free once a base scene exists.

Conventions
------------
* ``scene["objects"]`` holds floor + wall + small objects (see
  ``Holodeck.generate_scene``).
* ``scene["floor_objects"]``, ``scene["wall_objects"]`` and
  ``scene["small_objects"]`` are kept in sync with ``scene["objects"]`` so the
  scene remains internally consistent.
* Room vertices are stored in **meters**, object ``vertices`` in **cm**, and
  ``get_bbox_dims`` returns dimensions in **meters**.
"""

import copy
import math
import random
from typing import Any, Dict, List, Optional

from shapely.geometry import Point, Polygon, box

from ai2holodeck.generation.floor_objects import DFS_Solver_Floor
from ai2holodeck.generation.utils import get_annotations, get_bbox_dims

# Sub-lists of ``scene["objects"]`` that we keep in sync after mutating the
# placed objects. ``small_objects`` carries no ``object_name``/``vertices`` so
# it is matched by ``id`` only.
_PLACED_KEYS = ["floor_objects", "wall_objects", "small_objects"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _room_polygon_cm(room: Dict[str, Any]) -> Polygon:
    """Room polygon in centimetres (matching object ``vertices`` units)."""
    verts = [(x * 100, z * 100) for (x, z) in room["vertices"]]
    return Polygon(verts)


def _object_footprint_m(database: Dict[str, Any], asset_id: str) -> Dict[str, float]:
    """Bounding-box dims (m) for an asset."""
    return get_bbox_dims(database[asset_id])


def _resync_sublists(scene: Dict[str, Any]) -> None:
    """Re-derive ``floor_objects`` / ``wall_objects`` / ``small_objects`` from
    the current ``scene["objects"]`` so all four lists agree.

    Each placed object carries the same dict reference in every list, so we
    match by object identity (``id``). Small objects have ids like
    ``"name-0|receptacle-1 (room)"`` and never appear in floor/wall lists.
    """
    kept_ids = {o.get("id") for o in scene["objects"]}
    for key in _PLACED_KEYS:
        scene[key] = [o for o in scene.get(key, []) if o.get("id") in kept_ids]


def _recompute_bbox_vertices(
    center_x_cm: float, center_z_cm: float, dims_m: Dict[str, float], rotation_y: float
) -> List[List[float]]:
    """Axis-aligned bbox corners (cm) of an object of ``dims_m`` rotated by
    ``rotation_y`` degrees (multiples of 90) around ``(center_x_cm, center_z_cm)``.
    """
    x, _, z = dims_m["x"], dims_m["y"], dims_m["z"]
    if rotation_y % 180 == 0:
        dx, dz = x * 100 / 2.0, z * 100 / 2.0
    else:  # 90 / 270: swap footprint axes
        dx, dz = z * 100 / 2.0, x * 100 / 2.0
    return [
        [center_x_cm - dx, center_z_cm - dz],
        [center_x_cm - dx, center_z_cm + dz],
        [center_x_cm + dx, center_z_cm + dz],
        [center_x_cm + dx, center_z_cm - dz],
    ]


# ---------------------------------------------------------------------------
# variant 1 — keep ~half of the objects
# ---------------------------------------------------------------------------
def remove_half(scene: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    """Keep ``ceil(n/2)`` of the placed objects (seeded random subset).

    Non-object scene data (rooms, walls, doors, windows, lights, skybox) is
    left untouched.
    """
    scene = copy.deepcopy(scene)
    objects = scene.get("objects", [])
    n = len(objects)
    if n <= 1:
        return scene

    rng = random.Random(seed)
    keep_n = math.ceil(n / 2)
    kept = rng.sample(objects, keep_n)
    scene["objects"] = kept
    _resync_sublists(scene)
    return scene


# ---------------------------------------------------------------------------
# variant 2 — keep only the single biggest object
# ---------------------------------------------------------------------------
def keep_biggest_only(
    scene: Dict[str, Any], database: Dict[str, Any]
) -> Dict[str, Any]:
    """Keep only the placed object with the largest floor footprint (x*z)."""
    scene = copy.deepcopy(scene)
    objects = scene.get("objects", [])
    if not objects:
        return scene

    def footprint(o: Dict[str, Any]) -> float:
        try:
            d = _object_footprint_m(database, o["assetId"])
            return d["x"] * d["z"]
        except Exception:
            return -1.0

    best = max(objects, key=footprint)
    scene["objects"] = [best]
    _resync_sublists(scene)
    return scene


# ---------------------------------------------------------------------------
# variant 3 — scramble object positions within their rooms
# ---------------------------------------------------------------------------
def scramble_within_rooms(
    scene: Dict[str, Any],
    database: Dict[str, Any],
    seed: int = 0,
    max_attempts: int = 200,
) -> Dict[str, Any]:
    """Randomly relocate each placed object to a free grid point inside its
    own room (keeping the same rotation/dims, or a random 90° rotation when
    that helps the object fit). No LLM, no DFS solver, no Unity.

    Objects that cannot be placed inside their room within ``max_attempts``
    keep their original position. Small objects (no ``vertices`` / not in
    ``database``) are skipped.
    """
    scene = copy.deepcopy(scene)
    rng = random.Random(seed)

    rooms_by_id = {r["id"]: r for r in scene.get("rooms", [])}
    grid_cache: Dict[str, List[tuple]] = {}

    def grid_points(room: Dict[str, Any]) -> List[tuple]:
        rid = room["id"]
        if rid not in grid_cache:
            poly = _room_polygon_cm(room)
            rx = max(v[0] for v in room["vertices"]) - min(
                v[0] for v in room["vertices"]
            )
            rz = max(v[1] for v in room["vertices"]) - min(
                v[1] for v in room["vertices"]
            )
            grid_size = max(int(rx * 100 // 20), int(rz * 100 // 20), 1)
            solver = DFS_Solver_Floor(grid_size=grid_size)
            grid_cache[rid] = solver.create_grids(poly)
        return grid_cache[rid]

    placed_polys: List[Polygon] = []
    for obj in scene.get("objects", []):
        room = rooms_by_id.get(obj.get("roomId"))
        if room is None or "vertices" not in obj:
            continue
        asset_id = obj.get("assetId")
        if asset_id is None or asset_id not in database:
            continue
        dims = get_bbox_dims(database[asset_id])

        poly = _room_polygon_cm(room)
        points = grid_points(room)
        if not points:
            continue

        cur_rot = int(obj.get("rotation", {}).get("y", 0))
        rotations = [cur_rot] + [r for r in (0, 90, 180, 270) if r != cur_rot]
        rng.shuffle(rotations)

        chosen = None
        sample = rng.sample(points, min(max_attempts, len(points)))
        for pt in sample:
            for rot in rotations:
                verts = _recompute_bbox_vertices(pt[0], pt[1], dims, rot)
                obj_box = box(
                    min(verts[0][0], verts[2][0]),
                    min(verts[0][1], verts[2][1]),
                    max(verts[0][0], verts[2][0]),
                    max(verts[0][1], verts[2][1]),
                )
                if not poly.contains(obj_box):
                    continue
                if any(obj_box.intersects(p) for p in placed_polys):
                    continue
                chosen = (pt, rot, verts)
                break
            if chosen is not None:
                break

        if chosen is None:
            # relax: allow overlaps, just require inside the room
            for pt in sample:
                for rot in rotations:
                    verts = _recompute_bbox_vertices(pt[0], pt[1], dims, rot)
                    obj_box = box(
                        min(verts[0][0], verts[2][0]),
                        min(verts[0][1], verts[2][1]),
                        max(verts[0][0], verts[2][0]),
                        max(verts[0][1], verts[2][1]),
                    )
                    if poly.contains(obj_box):
                        chosen = (pt, rot, verts)
                        break
                if chosen is not None:
                    break

        if chosen is None:
            continue

        pt, rot, verts = chosen
        obj["position"] = {
            "x": pt[0] / 100.0,
            "y": dims["y"] / 2.0,
            "z": pt[1] / 100.0,
        }
        obj["rotation"] = {"x": 0, "y": rot, "z": 0}
        obj["vertices"] = verts
        placed_polys.append(Polygon(verts))

    return scene


# ---------------------------------------------------------------------------
# variant 4 — swap each object for the worst-matching valid asset
# ---------------------------------------------------------------------------
def _parse_object_type(object_name: str) -> str:
    """Strip the trailing ``-<index>`` the selector appends (e.g.
    ``white_bookshelf-0`` -> ``white_bookshelf``)."""
    if "-" in object_name:
        base, _, suffix = object_name.rpartition("-")
        if suffix.isdigit():
            return base
    return object_name


def select_worst_objects(scene: Dict[str, Any], holodeck) -> Dict[str, Any]:
    """For each placed floor/wall object, swap its ``assetId`` for the
    **lowest-scoring** retrieval candidate that still passes the selector's
    size + placement filters and fits within the original slot's footprint.

    Only CLIP+SBERT retrieval (local, free) is re-run — never the LLM, DFS, or
    Unity. Small objects are left untouched (they have no retrieval plan).
    """
    scene = copy.deepcopy(scene)
    selector = holodeck.object_selector
    retriever = holodeck.object_retriever
    database = holodeck.object_retriever.database
    plan = scene.get("object_selection_plan", {})
    rooms_by_id = {r["id"]: r for r in scene.get("rooms", [])}

    # cache per (object_type, description) -> filtered candidate list
    cache: Dict[tuple, List] = {}

    def candidates_for(
        obj_type: str, description: str, location: str, room: Dict[str, Any]
    ) -> List:
        key = (obj_type, description, location, room["id"])
        if key in cache:
            return cache[key]
        room_size = selector.get_room_size(room, scene["wall_height"])
        room_vertices = [(x * 100, y * 100) for (x, y) in room["vertices"]]
        cands = retriever.retrieve(
            [f"a 3D model of {obj_type}, {description}"],
            (
                selector.similarity_threshold_floor
                if location == "floor"
                else selector.similarity_threshold_wall
            ),
        )
        if location == "floor":
            cands = [
                c
                for c, ann in zip(
                    cands,
                    [get_annotations(database[c[0]]) for c in cands],
                )
                if ann["onFloor"]
                and not ann["onCeiling"]
                and all(
                    k not in ann["category"].lower()
                    for k in ["door", "window", "frame"]
                )
            ]
            cands = selector.check_object_size(cands, room_size)
            cands = selector.check_floor_placement(cands[:20], room_vertices, scene)
        else:
            cands = [c for c in cands if get_annotations(database[c[0]])["onWall"]]
            cands = [
                c
                for c in cands
                if "door" not in get_annotations(database[c[0]])["category"].lower()
            ]
            cands = [
                c
                for c in cands
                if "window" not in get_annotations(database[c[0]])["category"].lower()
            ]
            cands = selector.check_object_size(cands, room_size)
            cands = selector.check_thin_object(cands)
            cands = selector.check_wall_placement(cands[:20], room_vertices, scene)
        cache[key] = cands
        return cands

    swapped = 0
    for obj in scene.get("objects", []):
        obj_name = obj.get("object_name")
        asset_id = obj.get("assetId")
        room = rooms_by_id.get(obj.get("roomId"))
        if obj_name is None or asset_id is None or room is None:
            continue
        # small objects are not in the plan and have no vertices; skip them
        if "vertices" not in obj:
            continue
        obj_type = _parse_object_type(obj_name)
        room_plan = plan.get(room["roomType"], {})
        if obj_type not in room_plan:
            continue
        info = room_plan[obj_type]
        description = info.get("description", obj_type)
        location = info.get("location", "floor")
        if location not in ("floor", "wall"):
            location = "floor"

        try:
            cur_dims = get_bbox_dims(database[asset_id])
        except Exception:
            continue
        cur_footprint = cur_dims["x"] * cur_dims["z"]

        cands = candidates_for(obj_type, description, location, room)
        # exclude the current asset; keep only those that fit the slot's
        # footprint (so the worst object still roughly fits the room/area)
        viable = [c for c in cands if c[0] != asset_id and c[0] in database]
        viable_with_footprint = []
        for c in viable:
            try:
                d = get_bbox_dims(database[c[0]])
                if d["x"] * d["z"] <= cur_footprint + 1e-6:
                    viable_with_footprint.append((c, d))
            except Exception:
                continue
        if not viable_with_footprint:
            continue

        # candidates are sorted descending by score; worst = last
        worst_cand, worst_dims = viable_with_footprint[-1]
        new_id = worst_cand[0]

        # keep the same center + rotation, recompute bbox + y-height
        center_x_cm = obj["position"]["x"] * 100.0
        center_z_cm = obj["position"]["z"] * 100.0
        rot_y = int(obj.get("rotation", {}).get("y", 0))
        obj["assetId"] = new_id
        obj["position"]["y"] = worst_dims["y"] / 2.0
        obj["vertices"] = _recompute_bbox_vertices(
            center_x_cm, center_z_cm, worst_dims, rot_y
        )
        swapped += 1

    _resync_sublists(scene)
    return scene
