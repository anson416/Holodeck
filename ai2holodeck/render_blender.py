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


def _parse_bg_colors(s: str) -> list[tuple[int, int, int]]:
    """Parse one or more R,G,B triples separated by ';' — e.g. '255,0,0;0,255,0'."""
    s = s.strip()
    if not s or s.lower() == "none":
        return []
    out: list[tuple[int, int, int]] = []
    for chunk in s.split(";"):
        parts = [int(x) for x in chunk.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(f"bad bg color: {chunk!r} (expected R,G,B)")
        out.append((parts[0], parts[1], parts[2]))
    return out


def _composite_bg(alpha_png: str, out_png: str, color: tuple[int, int, int]) -> None:
    from PIL import Image  # local import so bpy-only runs don't require PIL globally
    img = Image.open(alpha_png).convert("RGBA")
    bg = Image.new("RGBA", img.size, (*color, 255))
    Image.alpha_composite(bg, img).convert("RGB").save(out_png)


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


def _parse_hdris(s: str) -> list[str]:
    """Comma-separated HDRI names. 'none' (or empty entry) means no HDRI."""
    return [x.strip() for x in s.split(",") if x.strip()]


def _build_scene(scene_data: dict, hide_ceiling: bool) -> None:
    bu.clear()
    bu.build_shell(scene_data, hide_ceiling=hide_ceiling)
    bu.place_objects(scene_data)
    bu.place_openings(scene_data)
    bu.add_lights(scene_data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--resolutions", type=_csv_ints, default=[512])
    parser.add_argument("--pitches", type=_csv_floats, default=[30.0])
    parser.add_argument("--yaws", type=_csv_floats, default=[0.0, 90.0, 180.0, 270.0])
    parser.add_argument("--focal", type=_csv_floats, default=[50.0])
    parser.add_argument(
        "--bg-color",
        type=_parse_bg_colors,
        default=[(128, 128, 128)],
        help="One or more R,G,B triples separated by ';' (e.g. '128,128,128;255,200,180'). Use 'none' to keep only the transparent render.",
    )
    parser.add_argument("--hdri", type=_parse_hdris, default=["none"], help="Comma-separated HDRI names (looked up under data/hdri/<name>.{exr,hdr}). Use 'none' for no HDRI; mix freely, e.g. 'none,city,sunset'.")
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
    hdri_specs: list[tuple[str, str | None]] = []  # (tag, path or None)
    for name in args.hdri:
        path = _resolve_hdri(name, repo_root)
        tag = "none" if path is None else Path(path).stem
        hdri_specs.append((tag, path))
    bg_colors: list[tuple[int, int, int]] = args.bg_color

    # Geometry + objects + JSON lights are built once. World (HDRI) is rebuilt
    # per HDRI inside the loop. Background color is composited via PIL after
    # rendering with film_transparent=True.
    _build_scene(scene_data, hide_ceiling=not args.show_ceiling)

    stem = scene_path.stem
    combos = list(itertools.product(args.resolutions, args.pitches, args.yaws, args.focal, args.fit_ratio, hdri_specs))
    print(f"[render] {len(combos)} geometry render(s), {max(1, len(bg_colors))} bg variant(s) each -> {out_dir}")
    for res, pitch, yaw, focal, fit, (hdri_tag, hdri_path) in combos:
        bu.set_world((0.5, 0.5, 0.5), hdri_path)
        bu.orbit_camera(
            scene_data,
            pitch_deg=pitch,
            yaw_deg=yaw,
            focal_mm=focal,
            fit_ratio=fit,
            aspect_ratio=1.0,
            cull_walls=not args.no_cull_walls,
        )
        base = (
            f"{stem}__res{res}_pitch{pitch:g}_yaw{yaw:g}"
            f"_focal{focal:g}_fit{fit:g}_hdri{hdri_tag}"
        )
        alpha_path = out_dir / f"{base}_alpha.png"
        print(f"[render] -> {alpha_path.name}")
        bu.render(str(alpha_path), resolution=res, samples=args.samples, engine=args.engine, transparent=True)

        if not bg_colors:
            continue
        for r, g, b in bg_colors:
            composed = out_dir / f"{base}_bg{r}-{g}-{b}.png"
            _composite_bg(str(alpha_path), str(composed), (r, g, b))
            print(f"[render]    + {composed.name}")


if __name__ == "__main__":
    main()
