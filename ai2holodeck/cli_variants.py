"""Variant-aware scene generation CLI.

Generates a base scene from a textual prompt, then (optionally) derives four
no-LLM-cost variants from it. Output layout::

    outputs/<UTC-timestamp>/
        config.json
        base/scene.json
        variant_01_half/scene.json          (--variants only)
        variant_02_biggest-only/scene.json  (--variants only)
        variant_03_scrambled/scene.json     (--variants only)
        variant_04_worst-object/scene.json  (--variants only)

Example::

    python ai2holodeck/cli_variants.py \
        --query "a living room" \
        --openai_api_key $OPENAI_API_KEY \
        --model_name gpt-4o \
        --temperature 0.7 \
        --variants
"""

import argparse
import datetime
import os
import sys

import compress_json

from ai2holodeck.constants import LLM_MODEL_NAME, OBJATHOR_ASSETS_DIR
from ai2holodeck.generation.holodeck import Holodeck
from ai2holodeck.generation import variants


def str2bool(v: str) -> bool:
    v = v.lower().strip()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"{v} cannot be converted to a bool")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a Holodeck scene from a text prompt, with optional no-LLM variants.",
    )
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
    p.add_argument("--model_name", default=LLM_MODEL_NAME, help="LLM model name.")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature. Omit for provider default.",
    )
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
    return p


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

    generate_image = str2bool(args.generate_image)

    holodeck = Holodeck(
        openai_api_key=args.openai_api_key,
        openai_org=args.openai_org,
        objaverse_asset_dir=OBJATHOR_ASSETS_DIR,
        single_room=str2bool(args.single_room),
        model_name=args.model_name,
        openai_api_base=args.openai_api_base,
        temperature=args.temperature,
    )

    # --- 1. generate the base scene (the only LLM spend) ------------------
    print(
        f"[generate] query={args.query!r} model={args.model_name} "
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

    # --- 2. lay out the run folder ----------------------------------------
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.output_dir, timestamp))
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "query": args.query,
        "model_name": args.model_name,
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
