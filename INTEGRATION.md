# VLM-Unreliability Evaluation Harness — Holodeck Integration

This fork of *Holodeck: Language Guided Generation of 3D Embodied AI Environments* is
integrated as one **target generator** in the VLM-evaluator reliability audit
(`vlm-unreliability`). Holodeck generates indoor scenes from text prompts; the added
harness renders them under a controlled sweep of camera/lighting factors and emits
content-perturbation variants. The audit (separate repo) consumes the rendered PNGs and
parses the swept factors out of the filenames — no Python import coupling.

## What was added / changed

| Path | Role |
|---|---|
| `ai2holodeck/constants.py` *(modified)* | Re-points the asset/feature/annotation paths at a local Objathor store. **These are machine-specific absolute paths** — adjust for your environment. |
| `ai2holodeck/generation/holodeck.py` *(modified)* | Drops the removed `OBJATHOR_VERSIONED_DIR` symbol from imports / path checks. |
| `ai2holodeck/render_blender.py`, `ai2holodeck/generation/blender_utils.py`, `data/hdri/*.exr` | Standalone Blender renderer + environment maps (already committed in this fork). |
| `generate_ablation_variants.py` | Content-perturbation variant generator (object removal + worst-match asset swap). |
| `run_vlmunr.sh` | Batch scene **generation** — 50 `main.py --query "…"` calls. |
| `create_renders_1a.sh … 1d.sh`, `create_renders_2.sh` | Per-factor render-sweep drivers (one factor each). |
| `hd_mr1.sh … hd_mr3.sh` | Older multi-room batch generation scripts. |

## Render CLI and filename scheme

`python ai2holodeck/render_blender.py --scene <scene.json> [factor flags]` sweeps the
Cartesian product of the requested factors and writes, per combo, a transparent master
plus one composite per background colour:

```
<stem>__res{res}_pitch{pitch}_yaw{yaw}_focal{focal}_fit{fit}_hdri{hdri}_alpha.png
<stem>__res{res}_pitch{pitch}_yaw{yaw}_focal{focal}_fit{fit}_hdri{hdri}_bg{r}-{g}-{b}.png
```

Token order: `res, pitch, yaw, focal, fit, hdri`; background is a `_bg{r}-{g}-{b}` suffix.
Here **pitch 90 = top-down** in this renderer's camera convention. Output dir defaults to
`<scene_dir>/renders/`.

## Swept factors (per `create_renders_*.sh`, each isolates one axis off the baseline)

| Driver | Factor | Levels |
|---|---|---|
| `create_renders_1a.sh` | resolution | 224, 256, 384, 448, 512, 640, 768, 1024 |
| `create_renders_1b.sh` | background | grays 0, 18, 65, 117, 128, 186, 204, 255 |
| `create_renders_1c.sh` | env map | city, sunset, interior, forest, sunrise, courtyard, night, studio |
| `create_renders_1d.sh` | focal length | 24, 35, 50, 85, 100, 200 |
| `create_renders_2.sh` | viewpoint | pitch {90, 60} × yaw {0, 30, …, 330} |

Baseline: res 512, bg 128, hdri city, focal 50, pitch 90, yaw 0, `--fit-ratio 1.0`.

## Content-perturbation variants (`generate_ablation_variants.py`)

```
python generate_ablation_variants.py --scene_path <scene.json> --output_dir <dir> \
    [--reduced_fractions 0.5 0.25 0.125] [--worst_match_indices 0 2 4]
```

- **Object removal** — `reduced_0p5`, `reduced_0p25`, `reduced_0p125`: keep a seeded
  random `round(n·frac)` of the top-level objects (layout/positions preserved).
- **Worst-match asset swap** — `worst_match_0`, `worst_match_2`, `worst_match_4`: for
  each object type, retrieve all candidate assets via CLIP+SBERT, sort worst-first, and
  swap in the asset at the given worst-rank index.

## Run

```bash
bash run_vlmunr.sh                                   # generate the 50 scenes
bash create_renders_1a.sh                            # … through create_renders_2.sh
python generate_ablation_variants.py --scene_path <scene.json> --output_dir <dir>
```

Requires Python 3.10 with `bpy`, the AI2-THOR / Objathor asset stack, and an OpenAI-
compatible LLM endpoint — none bundled. See the modified `constants.py` for the local
asset paths.
