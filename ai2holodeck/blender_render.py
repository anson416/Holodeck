"""Blender render driver for Holodeck scene JSONs.

Renders a scene (base or variant) to PNGs using the vendored ``bpa`` renderer,
following the mandated conventions:

* **Two-step alpha->composite**: render with a transparent film (HDRI still
  lights the scene), then composite the background colour over the alpha via
  ``bpa.Renderer.add_bg_to_rgba``. The transparent master keeps the env-only
  filename; the composited image gets ``_bg-<r>-<g>-<b>``.
* **``fit_ratio=1``** tight-fit framing (every visible vertex inside frustum).
* **Pitch convention**: pitch **0 = top-down**, ``rotation=(pitch, 0, yaw)``
  (the bpa convention). The filename always carries the literal pitch value.
* **Dollhouse walls**: camera-facing wall faces are transparent via back-face
  culling (set up in ``blender_utils.build_scene``), so the interior and far
  walls/doors/windows stay visible at oblique pitches and yaws.

Filename scheme (chosen to be consistent within Holodeck; the vlmunr scorer
parses it via a labelled-token regex)::

    transparent master:
      render_res-<r>_focal-<f>_pitch-<p>_yaw-<y>_env-<env>.png
    background composite:
      render_res-<r>_focal-<f>_pitch-<p>_yaw-<y>_env-<env>_bg-<r>-<g>-<b>.png

Outputs go into ``<scene_dir>/renderings/``.

This module imports ``bpy`` lazily-friendly (it imports at module load, so it
should only be imported when rendering is actually requested -- ``cli.py`` does
that only when ``--render`` / ``--render-all`` is passed).
"""

from __future__ import annotations

import os
from typing import Optional

from ai2holodeck.generation import bpa, blender_utils as bu

# isort: off
import bpy  # type: ignore  # noqa: F401  (imported for side effects / availability)

# isort: on


# --------------------------------------------------------------------------
# factor grids (paper Table 1, OFAT)
# --------------------------------------------------------------------------

RESOLUTIONS = [196, 224, 256, 336, 384, 448, 512, 768, 1024]
FOCAL_LENGTHS = [16, 24, 35, 50, 85, 100, 200]
PITCHES = [0, 15, 30, 45, 60, 75, 90]  # 0 == top-down (bpa convention)
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]
ENV_MAPS = [
    "city",
    "courtyard",
    "forest",
    "interior",
    "night",
    "studio",
    "sunrise",
    "sunset",
]
# Background sweep (grid 6). Note 118 is included per the explicit spec.
BACKGROUNDS = [
    (0, 0, 0),
    (65, 65, 65),
    (118, 118, 118),
    (128, 128, 128),
    (186, 186, 186),
    (204, 204, 204),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

WHITE_BG = (255, 255, 255)

# Baseline camera config (the only one that carries all 10 backgrounds).
BASELINE_RES = 512
BASELINE_FOCAL = 50
BASELINE_PITCH = 0  # top-down
BASELINE_YAW = 0
BASELINE_ENV = "city"
# The yaw sweep fixes pitch at 45 (oblique), per spec.
YAW_SWEEP_PITCH = 45


class RenderConfig:
    """A single camera configuration plus the backgrounds to composite."""

    __slots__ = ("res", "focal", "pitch", "yaw", "env", "backgrounds")

    def __init__(self, res, focal, pitch, yaw, env, backgrounds):
        self.res = int(res)
        self.focal = int(focal)
        self.pitch = int(pitch)
        self.yaw = int(yaw)
        self.env = env
        self.backgrounds = list(backgrounds)

    @property
    def key(self) -> tuple:
        return (self.res, self.focal, self.pitch, self.yaw, self.env)

    def master_filename(self) -> str:
        return (
            f"render_res-{self.res}_focal-{self.focal}"
            f"_pitch-{self.pitch}_yaw-{self.yaw}_env-{self.env}.png"
        )

    def composite_filename(self, bg: tuple[int, int, int]) -> str:
        r, g, b = bg
        return (
            f"render_res-{self.res}_focal-{self.focal}"
            f"_pitch-{self.pitch}_yaw-{self.yaw}_env-{self.env}"
            f"_bg-{r}-{g}-{b}.png"
        )


def _baseline_config(backgrounds) -> RenderConfig:
    return RenderConfig(
        BASELINE_RES,
        BASELINE_FOCAL,
        BASELINE_PITCH,
        BASELINE_YAW,
        BASELINE_ENV,
        backgrounds,
    )


def single_config() -> list[RenderConfig]:
    """The ``--render`` configuration: one baseline camera, white background."""
    return [_baseline_config([WHITE_BG])]


def render_all_configs() -> list[RenderConfig]:
    """The ``--render-all`` OFAT grids.

    Every camera config writes its transparent master plus a white-bg
    composite. The baseline camera config additionally carries all 10
    backgrounds (grid 6), with the white composite deduplicated.
    """
    base = dict(
        res=BASELINE_RES,
        focal=BASELINE_FOCAL,
        pitch=BASELINE_PITCH,
        yaw=BASELINE_YAW,
        env=BASELINE_ENV,
    )
    seen: set = set()
    out: list[RenderConfig] = []

    def add(**ov):
        c = {**base, **ov}
        k = (c["res"], c["focal"], c["pitch"], c["yaw"], c["env"])
        is_baseline = k == (
            BASELINE_RES,
            BASELINE_FOCAL,
            BASELINE_PITCH,
            BASELINE_YAW,
            BASELINE_ENV,
        )
        if k in seen:
            return
        seen.add(k)
        # Baseline camera gets all 10 backgrounds (white deduped inside).
        # Every other config gets only the white composite.
        bgs = BACKGROUNDS if is_baseline else [WHITE_BG]
        out.append(
            RenderConfig(c["res"], c["focal"], c["pitch"], c["yaw"], c["env"], bgs)
        )

    # Grid 1: resolution
    for r in RESOLUTIONS:
        add(res=r)
    # Grid 2: focal length
    for f in FOCAL_LENGTHS:
        add(focal=f)
    # Grid 3: pitch
    for p in PITCHES:
        add(pitch=p)
    # Grid 4: yaw (pitch fixed at 45)
    for y in YAWS:
        add(pitch=YAW_SWEEP_PITCH, yaw=y)
    # Grid 5: env map
    for e in ENV_MAPS:
        add(env=e)
    # Grid 6: backgrounds -- already materialised on the baseline config above;
    # nothing to add here (the baseline entry from grid 1 carries all 10 bgs).
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _hdri_path(env: str, hdri_dir: str) -> str:
    for ext in (".exr", ".hdr"):
        p = os.path.join(hdri_dir, f"{env}{ext}")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"HDRI for env '{env}' not found under {hdri_dir} (looked for .exr/.hdr)"
    )


def render_scene(
    scene_path: str,
    out_dir: str,
    configs: list[RenderConfig],
    hdri_dir: str,
    *,
    samples: int = 64,
    engine: str = "CYCLES",
    fit_ratio: float = 1.0,
) -> int:
    """Render ``scene_path`` for each config into ``out_dir/renderings``.

    Configs are grouped by env map: the world (HDRI) is rebuilt once per env,
    and within an env group the geometry is built once and reused across all
    camera configs (matching ``anchor_render.py``). Returns the # of PNGs
    written.
    """
    import compress_json

    scene = compress_json.load(scene_path)
    renderings_dir = os.path.join(out_dir, "renderings")
    os.makedirs(renderings_dir, exist_ok=True)

    # Group by env, preserving first-seen order.
    by_env: dict[str, list[RenderConfig]] = {}
    for c in configs:
        by_env.setdefault(c.env, []).append(c)

    n_written = 0
    for env, cfgs in by_env.items():
        hdri = _hdri_path(env, hdri_dir)
        # 1. fresh scene + geometry (walls are plain opaque; dollhouse occlusion
        #    is handled per-camera by bu.orbit_camera -> cull_near_walls).
        bpa.clear()
        bu.build_scene(scene)
        # 2. world/render setup after geometry.
        bpa.initialize(transparent=True, environment_map=(hdri, 1.0))
        sc = bpy.context.scene
        sc.render.engine = engine
        if engine == "CYCLES":
            sc.cycles.samples = max(1, samples)
            sc.cycles.device = "CPU"

        for c in cfgs:
            master = os.path.join(renderings_dir, c.master_filename())
            if not os.path.isfile(master):
                # orbit_camera hides near walls (cull_near_walls) THEN collects
                # visible vertices for tight-fit framing — the correct order for
                # the dollhouse convention.
                bu.orbit_camera(
                    scene,
                    pitch_deg=c.pitch,
                    yaw_deg=c.yaw,
                    focal_mm=float(c.focal),
                    fit_ratio=fit_ratio,
                    aspect_ratio=1.0,
                    cull_walls=True,
                )
                sc.render.resolution_x = c.res
                sc.render.resolution_y = c.res
                sc.render.filepath = master
                result = bpy.ops.render.render(write_still=True)
                if "FINISHED" not in result:
                    raise RuntimeError(f"render failed: {result}")
                n_written += 1
            # Composite backgrounds over the transparent master.
            for bg in c.backgrounds:
                comp = os.path.join(renderings_dir, c.composite_filename(bg))
                if not os.path.isfile(comp):
                    bpa.Renderer.add_bg_to_rgba(master, comp, color=bg)
                    n_written += 1
        print(f"[render] env={env}: {len(cfgs)} camera config(s) -> {renderings_dir}")
    return n_written
