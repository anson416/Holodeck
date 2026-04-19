"""Render a Holodeck scene JSON in Blender with configurable camera/lighting.

Usage (must run inside the `holodeck` conda env, which has `bpy` installed):

    conda run -n holodeck python ai2holodeck/render_blender.py \\
        --scene data/scenes/20250725-221049_a_bedroom/a_bedroom.json \\
        --resolutions 512 \\
        --pitches 30 \\
        --yaws 0,90,180,270 \\
        --focal 50 \\
        --bg-color 128,128,128 \\
        --hdri none

Comma-separated values in --resolutions/--pitches/--yaws/--focal are swept as the
cartesian product. Outputs go to <scene_dir>/renders/ unless --output-dir is set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

from ai2holodeck.generation import blender_utils as bu


def _csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_rgb(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--bg-color must be R,G,B")
    return tuple(parts)  # type: ignore[return-value]


def _resolve_hdri(name: str | None, repo_root: Path) -> str | None:
    if not name or name.lower() == "none":
        return None
    if os.path.isfile(name):
        return name
    for ext in (".exr", ".hdr"):
        cand = repo_root / "data" / "hdri" / f"{name}{ext}"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"HDRI not found: {name} (looked in data/hdri/)")


def _build_scene(scene_data: dict, bg_rgb: tuple[float, float, float], hdri_path: str | None, hide_ceiling: bool) -> None:
    bu.clear()
    bu.build_shell(scene_data, hide_ceiling=hide_ceiling)
    bu.place_objects(scene_data)
    bu.place_openings(scene_data)
    bu.add_lights(scene_data)
    bu.set_world(bg_rgb, hdri_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--resolutions", type=_csv_ints, default=[512])
    parser.add_argument("--pitches", type=_csv_floats, default=[30.0])
    parser.add_argument("--yaws", type=_csv_floats, default=[0.0, 90.0, 180.0, 270.0])
    parser.add_argument("--focal", type=_csv_floats, default=[50.0])
    parser.add_argument("--bg-color", type=_parse_rgb, default=(128, 128, 128))
    parser.add_argument("--hdri", type=str, default="none")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--fit-ratio", type=_csv_floats, default=[0.0], help="0..1 lerp between bounding-sphere and tight-fit camera distance (genxr-style).")
    parser.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE"], default="CYCLES")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--show-ceiling", action="store_true", help="Render the ceiling (default: hidden so interior is visible).")
    parser.add_argument("--no-cull-walls", action="store_true", help="Disable hiding walls between camera and scene center.")
    args = parser.parse_args()

    scene_path: Path = args.scene.resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    with open(scene_path) as f:
        scene_data = json.load(f)

    out_dir = args.output_dir or (scene_path.parent / "renders")
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    hdri_path = _resolve_hdri(args.hdri, repo_root)
    hdri_tag = "none" if hdri_path is None else Path(hdri_path).stem
    bg_r, bg_g, bg_b = args.bg_color
    bg_tag = f"{bg_r}-{bg_g}-{bg_b}"
    bg_rgb_norm = (bg_r / 255.0, bg_g / 255.0, bg_b / 255.0)

    # Geometry/world are independent of camera; build once per (bg,hdri) only.
    _build_scene(scene_data, bg_rgb_norm, hdri_path, hide_ceiling=not args.show_ceiling)

    stem = scene_path.stem
    combos = list(itertools.product(args.resolutions, args.pitches, args.yaws, args.focal, args.fit_ratio))
    print(f"[render] {len(combos)} render(s) -> {out_dir}")
    for res, pitch, yaw, focal, fit in combos:
        bu.orbit_camera(
            scene_data,
            pitch_deg=pitch,
            yaw_deg=yaw,
            focal_mm=focal,
            fit_ratio=fit,
            aspect_ratio=1.0,
            cull_walls=not args.no_cull_walls,
        )
        fname = (
            f"{stem}__res{res}_pitch{pitch:g}_yaw{yaw:g}"
            f"_focal{focal:g}_fit{fit:g}_bg{bg_tag}_hdri{hdri_tag}.png"
        )
        path = out_dir / fname
        print(f"[render] -> {path.name}")
        bu.render(str(path), resolution=res, samples=args.samples, engine=args.engine)


if __name__ == "__main__":
    main()
