# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Holodeck is a system for **language-guided generation of 3D embodied AI
environments**. Given a natural-language query (e.g. `"a living room"`), it
produces a complete AI2-THOR-compatible scene JSON: floor plan, walls, doors,
windows, materials, object placement, lighting, and a skybox. It uses an LLM
(OpenAI by default) for high-level layout decisions and CLIP + SBERT retrieval
over the Objathor / THOR asset databases for concrete asset selection.

Source: CVPR 2024 paper "Holodeck: Language Guided Generation of 3D Embodied
AI Environments" (Allen Institute for AI). See `README.md`.

## Tech Stack

- **Python 3.10** (`conda create -n holodeck python=3.10`)
- **LLM**: OpenAI via `openai==1.66.3` (default model in
  `ai2holodeck/constants.py:40`); prompts wrapped with `langchain==0.0.171`
- **Retrieval**: `open_clip_torch`, `sentence-transformers`, `torch==2.5.1`
- **Geometry / placement**: `shapely`, `rtree`, `scipy`, `cvxpy`,
  `gurobipy==10.0.3` (MILP solver, optional)
- **Simulator**: `ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897`
  (installed from the AI2 PyPI index, not in `requirements.txt`)
- **Asset pipeline**: `objathor`, `objaverse`
- **Serialization**: `compress_json`, `compress_pickle`

## Repository Layout

- `ai2holodeck/main.py` — CLI entry point. Three modes:
  `generate_single_scene`, `generate_multi_scenes`, `generate_variants`.
- `ai2holodeck/constants.py` — paths, env-var-overridable config, default LLM
  model, THOR commit pinning. **Note**: currently contains hardcoded absolute
  asset paths that override the `~/.objathor-assets` defaults — see lines
  17–21.
- `ai2holodeck/generation/holodeck.py` — `Holodeck` orchestrator class;
  `generate_scene` is the staged pipeline.
- `ai2holodeck/generation/` — one module per pipeline stage (`rooms.py`,
  `walls.py`, `doors.py`, `windows.py`, `object_selector.py`,
  `floor_objects.py`, `wall_objects.py`, `small_objects.py`,
  `ceiling_objects.py`, `lights.py`, `layers.py`, `skybox.py`), plus
  shared infrastructure (`objaverse_retriever.py`, `milp_utils.py`,
  `prompts.py`, `utils.py`, `empty_house.json`).
- `connect_to_unity.py` — loads a generated scene JSON into a running Unity
  AI2-THOR editor via the WSGI server.
- `data/` — generated scenes (`scenes/`, `scenes_mr1/`, `scenes_mr2/`,
  `scenes_mr3/`) and query lists.
- `hd_mr1.sh`, `hd_mr2.sh`, `hd_mr3.sh` — batch scripts that invoke
  `main.py` over many queries (currently use a non-default `--openai_api_base`
  and a Gemini model via an OpenAI-compatible proxy).

## Build / Run / Test

```bash
# Install (after creating the conda env)
make install              # pip install -e .
make install-dev          # pip install -r requirements.txt
python -m pip install --extra-index-url https://ai2thor-pypi.allenai.org \
    ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897

# One-time data download
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_assets             --version 2023_09_23
python -m objathor.dataset.download_annotations        --version 2023_09_23
python -m objathor.dataset.download_features           --version 2023_09_23

# Generate a scene
python ai2holodeck/main.py --query "a living room" --openai_api_key $OPENAI_API_KEY

# Generate from a query file
python ai2holodeck/main.py --mode generate_multi_scenes --query_file ./data/queries.txt

# Generate variants of an existing scene
python ai2holodeck/main.py --mode generate_variants --original_scene <PATH>

# Tests / formatting
make test                 # python -m pytest -x -s -v tests   (no `tests/` dir present yet)
make black                # python -m black .
```

Boolean CLI flags (`--generate_image`, `--use_milp`, `--add_ceiling`,
`--single_room`, …) are passed as strings (`"True"` / `"False"`) — see
`main.py:60-66`.

## Key Conventions

- **The pipeline mutates a single `scene` dict.** Each stage reads upstream
  keys and writes new ones. See `Holodeck.generate_scene`
  (`ai2holodeck/generation/holodeck.py:299`).
- **`used_assets` is set on the instance, not passed.** Each generator has a
  `self.used_assets` list assigned by the caller before its `generate_*`
  method runs.
- **Use DFS, not MILP**, for object placement unless you have a reason —
  README and code defaults agree (`--use_milp False`).
- **Add new paths and model names to `constants.py`**, not inline.
- **Add new prompts to `ai2holodeck/generation/prompts.py`** and wrap them in
  `langchain.PromptTemplate` inside the consuming generator's constructor.

## Additional Documentation

When the task touches the items below, consult the corresponding doc:

- `.claude/docs/architectural_patterns.md` — staged pipeline contract,
  generator class shape, LLM injection, CLIP+SBERT retrieval, DFS/MILP
  placement, retry-at-orchestrator pattern, CLI-boolean-as-string convention.
