"""Holodeck variant-aware scene generation CLI.

Generate a 3D embodied-AI scene (AI2-THOR-compatible JSON) from a natural-
language prompt, and optionally derive four no-LLM-cost variants from it.

Output layout (``--output_dir`` defaults to ``./outputs``)::

    outputs/<UTC-timestamp>/
        config.json
        base/scene.json
        variant_01_half/scene.json          (--variants only)
        variant_02_biggest-only/scene.json  (--variants only)
        variant_03_scrambled/scene.json     (--variants only)
        variant_04_worst-object/scene.json  (--variants only)

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

    python cli.py --query "a living room" \\
        --openai_api_key $OPENAI_API_KEY \\
        --model_name gpt-4o \\
        --temperature 0.7 \\
        --variants

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

   If you have the *entire* objathor bundle (assets + features +
   annotations) at a single flat root that is NOT laid out as
   ``<base>/<version>/...``, use ``--objathor_root <ROOT>`` instead — this
   maps ``<ROOT>/assets``, ``<ROOT>/features``, ``<ROOT>/annotations.json.gz``
   directly. (The Holodeck THOR data still needs ``--objathor_assets_base_dir``
   because it lives under a different layout.)

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
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a Holodeck scene from a text prompt, with optional no-LLM variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- prompt / LLM -----------------------------------------------------
    p.add_argument(
        "--query",
        "--prompt",
        dest="query",
        required=True,
        help="Textual scene description.",
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
        "--output_dir",
        default=os.path.join(os.getcwd(), "outputs"),
        help="Root directory for run folders.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed for the deterministic variants."
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
        help="Objathor assets version (default 2023_09_23). " "Maps to ASSETS_VERSION.",
    )
    res.add_argument(
        "--hd_base_version",
        default=None,
        help="Holodeck base data version (default 2023_09_23). "
        "Maps to HD_BASE_VERSION.",
    )
    res.add_argument(
        "--objathor_root",
        default=None,
        help="Flat objathor bundle root: <ROOT>/assets, "
        "<ROOT>/features, <ROOT>/annotations.json.gz. "
        "Maps to VLMUNR_OBJATHOR_ROOT.",
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
    if args.objathor_root:
        os.environ["VLMUNR_OBJATHOR_ROOT"] = args.objathor_root


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

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

    from ai2holodeck.constants import LLM_MODEL_NAME, OBJATHOR_ASSETS_DIR
    from ai2holodeck.generation.holodeck import Holodeck
    from ai2holodeck.generation import variants
    import compress_json

    model_name = args.model_name or LLM_MODEL_NAME
    generate_image = str2bool(args.generate_image)

    holodeck = Holodeck(
        openai_api_key=args.openai_api_key,
        openai_org=args.openai_org,
        objaverse_asset_dir=OBJATHOR_ASSETS_DIR,
        single_room=str2bool(args.single_room),
        model_name=model_name,
        openai_api_base=args.openai_api_base,
        temperature=args.temperature,
    )

    # --- 1. generate the base scene (the only LLM spend) ------------------
    print(
        f"[generate] query={args.query!r} model={model_name} "
        f"temperature={args.temperature}"
    )
    scene = holodeck.get_empty_scene()
    base_scene, _ = holodeck.generate_scene(
        scene=scene,
        query=args.query,
        save_dir=args.output_dir,
        generate_image=generate_image,
        generate_video=False,
        add_ceiling=str2bool(args.add_ceiling),
        add_time=False,  # we manage the folder name ourselves
        use_constraint=str2bool(args.use_constraint),
        use_milp=str2bool(args.use_milp),
        random_selection=str2bool(args.random_selection),
    )

    # --- 2. lay out the run folder ---------------------------------------
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
            "single_room": str2bool(args.single_room),
            "use_constraint": str2bool(args.use_constraint),
            "use_milp": str2bool(args.use_milp),
            "random_selection": str2bool(args.random_selection),
            "add_ceiling": str2bool(args.add_ceiling),
        },
        "resources": {
            "objathor_assets_base_dir": args.objathor_assets_base_dir,
            "assets_version": args.assets_version,
            "hd_base_version": args.hd_base_version,
            "objathor_root": args.objathor_root,
        },
        "seed": args.seed,
        "timestamp_utc": timestamp,
    }
    compress_json.dump(
        config, os.path.join(run_dir, "config.json"), json_kwargs=dict(indent=4)
    )

    def save(sub: str, sc) -> None:
        sub_dir = os.path.join(run_dir, sub)
        os.makedirs(sub_dir, exist_ok=True)
        compress_json.dump(
            sc, os.path.join(sub_dir, "scene.json"), json_kwargs=dict(indent=4)
        )

    # --- 3. base ----------------------------------------------------------
    save("base", base_scene)
    print(f"[done] base -> {os.path.join(run_dir, 'base', 'scene.json')}")

    # --- 4. variants (no LLM spend) --------------------------------------
    summary = [("base", len(base_scene.get("objects", [])))]
    if args.variants:
        database = holodeck.object_retriever.database

        half = variants.remove_half(base_scene, seed=args.seed)
        save("variant_01_half", half)
        summary.append(("variant_01_half", len(half.get("objects", []))))

        biggest = variants.keep_biggest_only(base_scene, database)
        save("variant_02_biggest-only", biggest)
        summary.append(("variant_02_biggest-only", len(biggest.get("objects", []))))

        scrambled = variants.scramble_within_rooms(base_scene, database, seed=args.seed)
        save("variant_03_scrambled", scrambled)
        summary.append(("variant_03_scrambled", len(scrambled.get("objects", []))))

        worst = variants.select_worst_objects(base_scene, holodeck)
        save("variant_04_worst-object", worst)
        summary.append(("variant_04_worst-object", len(worst.get("objects", []))))

    print("\n[summary]")
    width = max(len(name) for name, _ in summary)
    for name, n in summary:
        print(f"  {name:<{width}}  {n} objects")
    print(f"\nRun directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
