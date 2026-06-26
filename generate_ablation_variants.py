"""Generate ablation variants from an existing Holodeck scene JSON.

Produces two families of variants while preserving layout, positions, and
rotations exactly:

1. Reduced-object variants: randomly drop objects to a fraction of the
   original count (default 1/2, 1/4, 1/8).
2. Worst-match asset-swap variants: for each object type, rank all assets
   by combined CLIP + SBERT similarity to the type name and swap in assets
   from the bottom of the ranking.

Usage:
    python generate_ablation_variants.py \\
        --scene_path path/to/scene.json \\
        --output_dir path/to/variants
"""

import argparse
import copy
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import open_clip
from sentence_transformers import SentenceTransformer

from ai2holodeck.generation.objaverse_retriever import ObjathorRetriever
from ai2holodeck.generation.utils import get_bbox_dims


OBJECT_NAME_SUFFIX_RE = re.compile(r"-\d+$")


def object_name_from_obj(obj: dict) -> str:
    """Return the canonical object_name for an object dict.

    Top-level objects have an explicit `object_name` (e.g. 'dining_table-0').
    Small objects placed on receptacles instead carry an `id` like
    'low concrete decorative bowl-1|dining_table-2 (minimalist terrace)' —
    we strip the parent and room suffix.
    """
    if "object_name" in obj:
        return obj["object_name"]
    raw_id = obj.get("id", "")
    head = raw_id.split("|", 1)[0].split(" (", 1)[0]
    return head


def object_type_query(object_name: str) -> str:
    """Turn 'dining_table-0' into 'dining table' for semantic retrieval."""
    stem = OBJECT_NAME_SUFFIX_RE.sub("", object_name)
    return stem.replace("_", " ").strip()


def build_retriever() -> ObjathorRetriever:
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="laion2b_s32b_b82k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return ObjathorRetriever(
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
        sbert_model=sbert_model,
        retrieval_threshold=0,
    )


def filter_scene_by_ids(scene: dict, kept_ids: Set[str]) -> dict:
    """Return a deep copy of *scene* with only objects whose `id` is in kept_ids."""
    out = copy.deepcopy(scene)
    out["objects"] = [o for o in out["objects"] if o["id"] in kept_ids]

    for key in ("floor_objects", "wall_objects", "small_objects"):
        if key in out and isinstance(out[key], list):
            out[key] = [o for o in out[key] if o.get("id") in kept_ids]

    if isinstance(out.get("receptacle2small_objects"), dict):
        out["receptacle2small_objects"] = {
            rid: smalls
            for rid, smalls in out["receptacle2small_objects"].items()
            if rid in kept_ids
        }

    # selected_objects: dict[roomId] -> {"floor": [[name, assetId], ...], "wall": [...]}
    kept_names = {full_id.split(" (")[0] for full_id in kept_ids}
    if isinstance(out.get("selected_objects"), dict):
        for room, groups in out["selected_objects"].items():
            if not isinstance(groups, dict):
                continue
            for group_key, entries in groups.items():
                if isinstance(entries, list):
                    groups[group_key] = [
                        e for e in entries
                        if not (isinstance(e, list) and len(e) >= 1)
                        or e[0] in kept_names
                    ]

    return out


def make_reduced_variants(
    scene: dict, fractions: List[float], seed: int
) -> List[Tuple[str, dict]]:
    rng = random.Random(seed)
    objects = scene["objects"]
    n = len(objects)
    all_ids = [o["id"] for o in objects]

    variants = []
    for frac in fractions:
        keep_n = max(1, round(n * frac))
        kept_ids = set(rng.sample(all_ids, keep_n))
        variant = filter_scene_by_ids(scene, kept_ids)
        tag = f"reduced_{frac}".replace(".", "p")
        variants.append((tag, variant))
    return variants


def pick_worst_match_assets(
    retriever: ObjathorRetriever,
    object_types: Set[str],
    pick_indices: List[int],
    size_constrained: bool,
    type_to_target_size: Dict[str, List[float]],
) -> Dict[str, List[str]]:
    """For each type, return a list of assetIds corresponding to pick_indices in the
    worst-first ranking.
    """
    out: Dict[str, List[str]] = {}
    for obj_type in sorted(object_types):
        # threshold very low so all assets are returned
        results = retriever.retrieve([obj_type], threshold=-1e9)
        if not results:
            print(f"[warn] no candidates returned for type '{obj_type}'")
            out[obj_type] = []
            continue

        if size_constrained and obj_type in type_to_target_size:
            results = retriever.compute_size_difference(
                type_to_target_size[obj_type], results
            )

        # results are sorted descending by score; reverse for worst-first
        worst_first = list(reversed(results))
        picked = []
        for idx in pick_indices:
            if idx < len(worst_first):
                picked.append(worst_first[idx][0])
            else:
                picked.append(worst_first[-1][0])
        out[obj_type] = picked
    return out


def apply_asset_swap(
    scene: dict,
    variant_idx: int,
    worst_map: Dict[str, List[str]],
    database: dict,
    warn_clip_threshold: float = 0.2,
) -> dict:
    out = copy.deepcopy(scene)
    updated_name_to_asset: Dict[str, str] = {}

    for obj in out["objects"]:
        name = object_name_from_obj(obj)
        obj_type = object_type_query(name)
        picks = worst_map.get(obj_type, [])
        if variant_idx >= len(picks) or not picks[variant_idx]:
            continue
        new_asset_id = picks[variant_idx]
        old_asset_id = obj.get("assetId")

        old_bbox = get_bbox_dims(database[old_asset_id]) if old_asset_id in database else None
        new_bbox = get_bbox_dims(database[new_asset_id]) if new_asset_id in database else None

        obj["assetId"] = new_asset_id
        updated_name_to_asset[name] = new_asset_id

        if old_bbox and new_bbox:
            dx = abs(new_bbox["x"] - old_bbox["x"]) / max(old_bbox["x"], 1e-6)
            dz = abs(new_bbox["z"] - old_bbox["z"]) / max(old_bbox["z"], 1e-6)
            if dx > warn_clip_threshold or dz > warn_clip_threshold:
                print(
                    f"[warn] variant {variant_idx} '{obj['id']}': "
                    f"new bbox differs (dx={dx:.0%}, dz={dz:.0%}) — may clip"
                )

    # mirror updates onto floor/wall/small object lists
    for key in ("floor_objects", "wall_objects", "small_objects"):
        if key in out and isinstance(out[key], list):
            for o in out[key]:
                new_asset = updated_name_to_asset.get(o.get("object_name"))
                if new_asset:
                    o["assetId"] = new_asset

    # selected_objects: update per-room asset id lists
    if isinstance(out.get("selected_objects"), dict):
        for _room, groups in out["selected_objects"].items():
            if not isinstance(groups, dict):
                continue
            for _group, entries in groups.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if isinstance(entry, list) and len(entry) >= 2:
                        name = entry[0]
                        if name in updated_name_to_asset:
                            entry[1] = updated_name_to_asset[name]

    return out


def make_worst_match_variants(
    scene: dict,
    retriever: ObjathorRetriever,
    pick_indices: List[int],
    size_constrained: bool,
) -> List[Tuple[str, dict]]:
    type_to_target_size: Dict[str, List[float]] = {}
    object_types: Set[str] = set()
    for obj in scene["objects"]:
        name = object_name_from_obj(obj)
        t = object_type_query(name)
        object_types.add(t)
        if t not in type_to_target_size and obj.get("assetId") in retriever.database:
            bbox = get_bbox_dims(retriever.database[obj["assetId"]])
            type_to_target_size[t] = [bbox["x"] * 100, bbox["y"] * 100, bbox["z"] * 100]

    worst_map = pick_worst_match_assets(
        retriever, object_types, pick_indices, size_constrained, type_to_target_size
    )

    variants = []
    for i, idx in enumerate(pick_indices):
        variant = apply_asset_swap(scene, i, worst_map, retriever.database)
        variants.append((f"worst_match_{idx}", variant))
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reduced_fractions", type=float, nargs="+", default=[0.5, 0.25, 0.125]
    )
    parser.add_argument(
        "--worst_match_indices", type=int, nargs="+", default=[0, 2, 4]
    )
    parser.add_argument("--size_constrained", action="store_true")
    parser.add_argument(
        "--skip_worst_match", action="store_true",
        help="Skip CLIP/SBERT loading and emit only reduced-object variants.",
    )
    args = parser.parse_args()

    with open(args.scene_path) as f:
        scene = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.scene_path))[0]

    print(f"Scene has {len(scene['objects'])} objects")

    reduced = make_reduced_variants(scene, args.reduced_fractions, args.seed)
    for tag, variant in reduced:
        out_path = os.path.join(args.output_dir, f"{base_name}_{tag}.json")
        with open(out_path, "w") as f:
            json.dump(variant, f)
        print(f"  wrote {out_path} (n={len(variant['objects'])})")

    if args.skip_worst_match:
        return

    print("Loading CLIP + SBERT for worst-match asset swap...")
    retriever = build_retriever()

    swaps = make_worst_match_variants(
        scene, retriever, args.worst_match_indices, args.size_constrained
    )
    for tag, variant in swaps:
        out_path = os.path.join(args.output_dir, f"{base_name}_{tag}.json")
        with open(out_path, "w") as f:
            json.dump(variant, f)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
