"""Holodeck variant-aware scene generation CLI.

Generate a 3D embodied-AI scene (AI2-THOR-compatible JSON) from a natural-
language prompt, and optionally derive four no-LLM-cost variants from it.

Output layout (``--output_dir`` defaults to ``./outputs``)::

    outputs/<UTC-timestamp>/
        config.json
        base/
            scene.json
            renderings/            # --generate_image True (Unity top-down.png)
                                  # and/or --render / --render_all (Blender)
            meshes/                # only when --export_meshes True
        variant_01_half/           # --variants only
            scene.json
            renderings/            # --render / --render_all (Blender)
            meshes/                # only when --export_meshes True
        variant_02_biggest-only/   # --variants only
            scene.json
        variant_03_scrambled/      # --variants only
            scene.json
        variant_04_worst-object/   # --variants only
            scene.json

Notes on the layout:
  * ``scene.json`` is the AI2-THOR-compatible scene. (Its name is illustrative;
    it is literally written as ``scene.json``.)
  * ``renderings/`` holds rendered images. The folder name **must** be
    ``renderings`` whenever it exists; this is enforced by the CLI.
    ``--generate_image`` writes the Unity top-down as ``renderings/top-down.png``;
    ``--render`` / ``--render_all`` write Blender renders as
    ``renderings/render_res-<r>_focal-<f>_pitch-<p>_yaw-<y>_env-<env>.png``
    (transparent master) and ``..._env-<env>_bg-<r>-<g>-<b>.png`` (background
    composite). The two pipelines are independent and may both be on.
  * ``meshes/`` holds each placed object's 3D mesh + textures, exported as
    ``meshes/<assetId>/<assetId>.glb`` (+ ``albedo.jpg`` etc.). Populated only
    with ``--export_meshes``; omitted otherwise.

Rendering (Blender) — two mutually-exclusive flags:
  * ``--render``       - one config per scene: 512x512, white bg, 50mm, pitch 0
                         (top-down), yaw 0, env "city". Transparent master +
                         white-bg composite per scene.
  * ``--render_all``   - the full OFAT grid (resolution / focal / pitch / yaw /
                         env / background sweeps). See
                         ``ai2holodeck/blender_render.py``.

Rendering uses the vendored ``bpa`` renderer with the mandated conventions:
two-step transparent-then-composite (HDRI still lights the transparent pass),
``fit_ratio=1`` tight-fit framing, pitch 0 == top-down, and the dollhouse wall
convention (camera-facing wall faces transparent via back-face culling, so the
interior + far walls/doors/windows stay visible). HDRI files live under
``<repo>/hdri/`` (override with ``--hdri_dir``).

Two invocation modes (exactly one of ``--prompt`` / ``--path``):
  * ``--prompt``  - generate the base scene first, then render base (+ variants
                    if ``--variants``). ``--render`` is independent of
                    ``--generate_image`` (Unity).
  * ``--path``    - skip generation; render every ``base/`` and ``variant_*/``
                    ``scene.json`` already present under the given
                    ``outputs/<datetime>/`` directory. Re-runnable without the
                    LLM/objathor assets/Unity.

The four variants mutate only ``scene["objects"]`` (placed floor + wall +
small objects). They never re-run the LLM, the DFS placement solver, or the
AI2-THOR / Unity controller, so they are essentially free once a base scene
exists:
  * half           - keep a seeded ``ceil(n/2)`` random subset of objects
  * biggest-only   - keep only the single largest object (by footprint)
  * scrambled      - relocate each object to a random free grid point inside
                     its own room (within-room shuffle)
  * worst-object   - swap each placed asset for the lowest-scoring retrieval
                     candidate that still fits the slot (CLIP+SBERT retrieval
                     only, never the LLM)

Example
-------
::

    # generate + variants + Blender render
    python cli.py --query "a living room" \\
        --openai_api_key $OPENAI_API_KEY \\
        --model_name gpt-4o \\
        --temperature 0.7 \\
        --variants \\
        --render

    # re-render an existing run with the full sweep (no LLM)
    python cli.py --path outputs/20260708-120000 --render_all

============================================================================
EXTERNAL LOCAL RESOURCES REQUIRED
============================================================================

Holodeck is NOT self-contained: generation needs several large local data
bundles plus an LLM endpoint. They are listed below in order of importance.

1. Objathor asset database (3D models, features, annotations)
   -----------------------------------------------------------
   This is the big one (~tens of GB). It contains the AI2-THOR / Objaverse
   3D models, their CLIP + SBERT feature vectors, and annotation metadata.
   Holodeck cannot select or place any object without it.

   The bundle is organised as::

       <ROOT>/
           assets/                 # 3D model asset files (OBJ/USDZ/etc.)
           features/               # clip_features.pkl, sbert_features.pkl
           annotations.json.gz     # per-asset metadata (onFloor/onWall/...)
       <OBJATHOR_ASSETS_BASE_DIR>/holodeck/<HD_BASE_VERSION>/
           thor_object_data/
               clip_features.pkl   # THOR-native object features
               sbert_features.pkl
               annotations.json.gz

   How to obtain it (run once, ~tens of GB download)::

       conda activate holodeck
       python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
       python -m objathor.dataset.download_assets             --version 2023_09_23
       python -m objathor.dataset.download_annotations       --version 2023_09_23
       python -m objathor.dataset.download_features          --version 2023_09_23

   By default everything lands under ``~/.objathor-assets/``. If you
   downloaded it elsewhere (or copied it from another machine), point this
   CLI at it with:

       --objathor_assets_base_dir <DIR>   (default ~/.objathor-assets)

   ``<DIR>`` must contain ``<ASSETS_VERSION>/assets``, ``.../features``,
   ``.../annotations.json.gz`` AND ``holodeck/<HD_BASE_VERSION>/...``.
   Override the versions with ``--assets_version`` / ``--hd_base_version``
   if you used a non-default version.

2. AI2-THOR Unity executable
   --------------------------
   Needed for small-object generation and for ``--generate_image True`` /
   video. The ``ai2thor`` package auto-downloads the Unity binary on first
   use into ``~/.ai2thor/``. No manual setup on macOS; on Linux you also
   need::

       apt-get -y install libvulkan1 pciutils xserver-xorg
       ai2thor-xorg start

   Install ai2thor itself with the pinned commit::

       python -m pip install --extra-index-url https://ai2thor-pypi.allenai.org \\
           ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897

   If you have no GPU/display or just want the JSON without rendering, set
   ``--generate_image False`` (the default). Small-object generation still
   spawns Unity inside ``generate_scene``; if that fails it is caught and
   skipped, leaving an otherwise-valid scene.

3. CLIP + SBERT model weights (downloaded automatically on first run)
   -------------------------------------------------------------------
   ``open_clip`` fetches ``ViT-L-14`` (``laion2b_s32b_b82k``) and
   ``sentence-transformers`` fetches ``all-mpnet-base-v2``. They cache in
   ``~/.cache/huggingface/`` (or ``$HF_HOME``). Just make sure the first
   run has internet access. To relocate the cache, set ``HF_HOME=<DIR>``
   in your environment.

4. An LLM endpoint
   ----------------
   Holodeck calls a chat-completion LLM for floor plans, door/window plans,
   object selection, and placement constraints. Provide:

       --openai_api_key <KEY>             (or set OPENAI_API_KEY)
       --openai_api_base <URL>            (OpenAI-compatible proxy, optional)
       --model_name <MODEL>               (default from constants.py)
       --temperature <FLOAT>              (optional; omitted = provider default)

   The default model in ``constants.py`` may be pinned to a snapshot you do
   not have access to — pass ``--model_name`` explicitly (e.g. ``gpt-4o``).

Quick sanity check (no LLM, no Unity): the variant functions in
``ai2holodeck.generation.variants`` can be exercised on any existing scene
under ``data/scenes/``.
"""

import argparse
import datetime
import os
import shutil
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a Holodeck scene from a text prompt, with optional no-LLM variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- prompt / path (exactly one required) -----------------------------
    p.add_argument(
        "--query",
        "--prompt",
        dest="query",
        default=None,
        help="Textual scene description. Exactly one of --prompt / --path is "
        "required. With --prompt the scene is generated first; --render then "
        "renders the freshly-generated scenes.",
    )
    p.add_argument(
        "--path",
        dest="path",
        default=None,
        help="An existing outputs/<datetime>/ run directory to render in place "
        "(no generation). Exactly one of --prompt / --path is required. With "
        "--path and --render/--render-all, every base/ and variant_*/scene.json "
        "under the path is rendered into its own renderings/ folder.",
    )
    p.add_argument(
        "--openai_api_key",
        default=None,
        help="LLM API key. Falls back to OPENAI_API_KEY env var.",
    )
    p.add_argument(
        "--openai_api_base", default=None, help="OpenAI-compatible base URL."
    )
    p.add_argument(
        "--openai_org",
        default=None,
        help="OpenAI org string. Falls back to OPENAI_ORG env var.",
    )
    p.add_argument(
        "--model_name",
        default=None,
        help="LLM model name. Defaults to constants.LLM_MODEL_NAME.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature. Omit for provider default.",
    )

    # --- variants / generation flags --------------------------------------
    p.add_argument(
        "--variants",
        action="store_true",
        help="Also generate the four variants (half / biggest-only / scrambled / worst-object).",
    )
    p.add_argument(
        "--generate_image",
        type=str,
        default="False",
        help="Render a top-down image of each scene (needs Unity).",
    )
    p.add_argument(
        "--single_room", type=str, default="False", help="Generate a single-room scene."
    )
    p.add_argument(
        "--use_constraint",
        type=str,
        default="True",
        help="Use LLM constraints for object placement.",
    )
    p.add_argument(
        "--use_milp",
        type=str,
        default="False",
        help="Use MILP for the placement solver.",
    )
    p.add_argument(
        "--random_selection",
        type=str,
        default="False",
        help="Use more-random object selection.",
    )
    p.add_argument(
        "--add_ceiling", type=str, default="False", help="Add ceiling objects."
    )
    p.add_argument(
        "--export_meshes",
        action="store_true",
        help="Copy each placed object's .glb mesh + textures into a "
        "``meshes/`` folder under every scene dir. Off by default (it can be "
        "large).",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join(os.getcwd(), "outputs"),
        help="Root directory for run folders.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed for the deterministic variants."
    )

    # --- rendering (Blender) ----------------------------------------------
    render_grp = p.add_argument_group("rendering (Blender)")
    render_mx = render_grp.add_mutually_exclusive_group()
    render_mx.add_argument(
        "--render",
        action="store_true",
        help="Render each scene (base + variants) in Blender with one fixed "
        "configuration: 512x512, white bg, 50mm, pitch 0 (top-down), yaw 0, "
        "env 'city'. Writes a transparent master + a white-bg composite per "
        "scene into renderings/. Independent of --generate_image (Unity).",
    )
    render_mx.add_argument(
        "--render_all",
        action="store_true",
        help="Render each scene across the full OFAT grid (resolution, focal, "
        "pitch, yaw, env, background sweeps). See ai2holodeck/blender_render.py.",
    )
    render_grp.add_argument(
        "--hdri_dir",
        default=None,
        help="Directory with <env>.exr HDRI files (e.g. city.exr). Defaults "
        "to <repo>/hdri.",
    )
    render_grp.add_argument(
        "--render_samples",
        type=int,
        default=64,
        help="Cycles samples per render (lower = faster, noisier).",
    )
    render_grp.add_argument(
        "--render_engine",
        choices=["CYCLES", "BLENDER_EEVEE"],
        default="CYCLES",
        help="Blender render engine.",
    )

    # --- external resource overrides (set as env vars before import) ----
    res = p.add_argument_group("external resources")
    res.add_argument(
        "--objathor_assets_base_dir",
        default=None,
        help="Base dir holding <version>/assets, features, "
        "annotations AND holodeck/<hd_version>/. "
        "Maps to OBJATHOR_ASSETS_BASE_DIR (default ~/.objathor-assets).",
    )
    res.add_argument(
        "--assets_version",
        default=None,
        help="Objathor assets version (default 2023_09_23). Maps to ASSETS_VERSION.",
    )
    res.add_argument(
        "--hd_base_version",
        default=None,
        help="Holodeck base data version (default 2023_09_23). "
        "Maps to HD_BASE_VERSION.",
    )
    return p


def str2bool(v: str) -> bool:
    v = v.lower().strip()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"{v} cannot be converted to a bool")


def _apply_resource_env(args) -> None:
    """constants.py derives all asset paths from env vars at import time, so
    these must be set BEFORE the holodeck package is imported."""
    if args.objathor_assets_base_dir:
        os.environ["OBJATHOR_ASSETS_BASE_DIR"] = args.objathor_assets_base_dir
    if args.assets_version:
        os.environ["ASSETS_VERSION"] = args.assets_version
    if args.hd_base_version:
        os.environ["HD_BASE_VERSION"] = args.hd_base_version


def _query_name(query: str) -> str:
    """Mirror ``Holodeck.generate_scene``'s folder/file naming so we can
    locate the file it dumps, then rename it to ``scene.json``."""
    q = query.replace("_", " ")
    return q.replace(" ", "_").replace("'", "")[:30]


def _organize_renderings(scene_dir: str) -> None:
    """Move any ``render.png`` left by ``generate_scene`` into a folder named
    exactly ``renderings``. The folder name is a hard requirement."""
    render_png = os.path.join(scene_dir, "render.png")
    if os.path.exists(render_png):
        renderings_dir = os.path.join(scene_dir, "renderings")
        os.makedirs(renderings_dir, exist_ok=True)
        shutil.move(render_png, os.path.join(renderings_dir, "top-down.png"))


def _rename_to_scene_json(scene_dir: str, query_name: str) -> str:
    """Rename ``generate_scene``'s ``<query_name>.json`` dump to ``scene.json``.
    Returns the final ``scene.json`` path (or the original path if the rename
    was not needed)."""
    src = os.path.join(scene_dir, f"{query_name}.json")
    dst = os.path.join(scene_dir, "scene.json")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.move(src, dst)
    return dst


def _export_meshes(scene, assets_dir: str, meshes_dir: str) -> int:
    """Copy each placed object's 3D mesh + textures into ``meshes/<assetId>/``.
    Returns the number of unique assets exported."""
    os.makedirs(meshes_dir, exist_ok=True)
    seen = set()
    exported = 0
    for obj in scene.get("objects", []):
        aid = obj.get("assetId")
        if not aid or aid in seen:
            continue
        seen.add(aid)
        src = os.path.join(assets_dir, aid)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(meshes_dir, aid)
        os.makedirs(dst, exist_ok=True)
        for fn in os.listdir(src):
            s = os.path.join(src, fn)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(dst, fn))
        exported += 1
    return exported


def _discover_scene_dirs(run_dir: str) -> list[str]:
    """Find every ``<run_dir>/<scene>/scene.json`` (base + variant_*)."""
    out = []
    if not os.path.isdir(run_dir):
        return out
    for name in sorted(os.listdir(run_dir)):
        scene_json = os.path.join(run_dir, name, "scene.json")
        if os.path.isfile(scene_json):
            out.append(os.path.join(run_dir, name))
    return out


def _render_scenes(args, scene_dirs: list[str]) -> None:
    """Render each scene dir with Blender (``--render`` / ``--render_all``)."""
    if not (args.render or args.render_all):
        return
    from ai2holodeck.blender_render import (
        render_all_configs,
        render_scene,
        single_config,
    )

    configs = render_all_configs() if args.render_all else single_config()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    hdri_dir = args.hdri_dir or os.path.join(repo_root, "hdri")
    print(
        f"[render] {len(scene_dirs)} scene(s) x {len(configs)} camera config(s) "
        f"-> each scene's renderings/"
    )
    for sd in scene_dirs:
        scene_json = os.path.join(sd, "scene.json")
        n = render_scene(
            scene_json,
            sd,
            configs,
            hdri_dir,
            samples=args.render_samples,
            engine=args.render_engine,
        )
        print(
            f"[render] {os.path.basename(sd)}: {n} PNG(s) -> {os.path.join(sd, 'renderings')}"
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Exactly one of --prompt / --path is required.
    if bool(args.query) == bool(args.path):
        print(
            "[ERROR] Pass exactly one of --prompt/--query (generate+render) or "
            "--path (render existing scenes only).",
            file=sys.stderr,
        )
        return 2
    do_render = bool(args.render or args.render_all)

    # --path mode: no generation, no LLM key, no objathor assets needed -- only
    # Blender rendering of scenes already on disk.
    if args.path is not None:
        if not do_render:
            print(
                "[ERROR] --path requires --render or --render_all "
                "(generation is not available in --path mode).",
                file=sys.stderr,
            )
            return 2
        run_dir = os.path.abspath(args.path)
        scene_dirs = _discover_scene_dirs(run_dir)
        if not scene_dirs:
            print(f"[ERROR] No <dir>/scene.json found under {run_dir}", file=sys.stderr)
            return 2
        _render_scenes(args, scene_dirs)
        return 0

    # --- --prompt mode: generation -----------------------------------------
    if args.openai_api_key is None:
        args.openai_api_key = os.environ.get("OPENAI_API_KEY")
    if args.openai_org is None:
        args.openai_org = os.environ.get("OPENAI_ORG")
    if args.openai_api_key is None:
        print(
            "[ERROR] No API key provided (use --openai_api_key or set OPENAI_API_KEY).",
            file=sys.stderr,
        )
        return 2

    # Import only AFTER resource env vars are set, so constants.py picks
    # up the overridden paths.
    _apply_resource_env(args)

    import compress_json

    from ai2holodeck.constants import LLM_MODEL_NAME, OBJATHOR_ASSETS_DIR
    from ai2holodeck.generation import variants
    from ai2holodeck.generation.holodeck import Holodeck

    model_name = args.model_name or LLM_MODEL_NAME
    generate_image = str2bool(args.generate_image)

    # --- 1. lay out the run folder FIRST so generate_scene writes into it --
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.output_dir, timestamp))
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "query": args.query,
        "model_name": model_name,
        "openai_api_base": args.openai_api_base,
        "openai_api_key_provided": args.openai_api_key is not None,
        "temperature": args.temperature,
        "flags": {
            "variants": args.variants,
            "generate_image": generate_image,
            "export_meshes": args.export_meshes,
            "single_room": str2bool(args.single_room),
            "use_constraint": str2bool(args.use_constraint),
            "use_milp": str2bool(args.use_milp),
            "random_selection": str2bool(args.random_selection),
            "add_ceiling": str2bool(args.add_ceiling),
            "render": args.render,
            "render_all": args.render_all,
            "render_samples": args.render_samples,
            "render_engine": args.render_engine,
        },
        "resources": {
            "objathor_assets_base_dir": args.objathor_assets_base_dir,
            "assets_version": args.assets_version,
            "hd_base_version": args.hd_base_version,
        },
        "seed": args.seed,
        "timestamp_utc": timestamp,
    }
    compress_json.dump(
        config, os.path.join(run_dir, "config.json"), json_kwargs=dict(indent=4)
    )

    holodeck = Holodeck(
        openai_api_key=args.openai_api_key,
        openai_org=args.openai_org,
        objaverse_asset_dir=OBJATHOR_ASSETS_DIR,
        single_room=str2bool(args.single_room),
        model_name=model_name,
        openai_api_base=args.openai_api_base,
        temperature=args.temperature,
    )

    # --- 2. generate the base scene (the only LLM spend) ------------------
    print(
        f"[generate] query={args.query!r} model={model_name} "
        f"temperature={args.temperature}"
    )
    qname = _query_name(args.query)
    base_scene, base_dir = holodeck.generate_scene(
        scene=holodeck.get_empty_scene(),
        query=args.query,
        save_dir=run_dir,
        folder_name="base",  # write directly into run_dir/base
        generate_image=generate_image,
        generate_video=False,
        add_ceiling=str2bool(args.add_ceiling),
        add_time=False,  # use folder_name verbatim
        use_constraint=str2bool(args.use_constraint),
        use_milp=str2bool(args.use_milp),
        random_selection=str2bool(args.random_selection),
    )
    # generate_scene dumps <qname>.json + render.png; normalise the layout.
    _rename_to_scene_json(base_dir, qname)
    _organize_renderings(base_dir)
    mesh_n = 0
    if args.export_meshes:
        mesh_n = _export_meshes(
            base_scene, OBJATHOR_ASSETS_DIR, os.path.join(base_dir, "meshes")
        )
    print(f"[done] base -> {os.path.join(base_dir, 'scene.json')}")
    if generate_image and os.path.isdir(os.path.join(base_dir, "renderings")):
        print(f"[done] base renderings -> {os.path.join(base_dir, 'renderings')}")
    if args.export_meshes:
        print(f"[done] base meshes -> {mesh_n} assets")

    # --- 3. variants (no LLM spend) --------------------------------------
    summary = [("base", len(base_scene.get("objects", [])))]
    if args.variants:
        database = holodeck.object_retriever.database
        variant_jobs = [
            ("variant_01_half", variants.remove_half(base_scene, seed=args.seed)),
            (
                "variant_02_biggest-only",
                variants.keep_biggest_only(base_scene, database),
            ),
            (
                "variant_03_scrambled",
                variants.scramble_within_rooms(base_scene, database, seed=args.seed),
            ),
            (
                "variant_04_worst-object",
                variants.select_worst_objects(base_scene, holodeck),
            ),
        ]
        for name, vscene in variant_jobs:
            vdir = os.path.join(run_dir, name)
            os.makedirs(vdir, exist_ok=True)
            compress_json.dump(
                vscene, os.path.join(vdir, "scene.json"), json_kwargs=dict(indent=4)
            )
            if args.export_meshes:
                m = _export_meshes(
                    vscene, OBJATHOR_ASSETS_DIR, os.path.join(vdir, "meshes")
                )
                print(f"[done] {name} meshes -> {m} assets")
            summary.append((name, len(vscene.get("objects", []))))
            print(f"[done] {name} -> {os.path.join(vdir, 'scene.json')}")

    print("\n[summary]")
    width = max(len(name) for name, _ in summary)
    for name, n in summary:
        print(f"  {name:<{width}}  {n} objects")
    print(f"\nRun directory: {run_dir}")

    # --- 4. Blender rendering (optional; no LLM/Unity) --------------------
    if do_render:
        scene_dirs = _discover_scene_dirs(run_dir)
        _render_scenes(args, scene_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
