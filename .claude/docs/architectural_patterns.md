# Architectural Patterns

Patterns observed across multiple files in `ai2holodeck/generation/`.

## 1. Staged Pipeline on a Mutable Scene Dict

`Holodeck.generate_scene` (`ai2holodeck/generation/holodeck.py:299`) is a fixed
sequence of stages, each reading and writing keys on a single `scene: Dict`:

`empty_house → generate_rooms → generate_walls → generate_doors →
generate_windows → select_objects → floor_objects → wall_objects →
small_objects → (ceiling_objects) → lights → map_asset2layer → getSkybox →
change_ceiling_material`.

Each stage adds new top-level keys (e.g. `rooms`, `walls`, `doors`,
`selected_objects`, `floor_objects`, `objects`, `proceduralParameters.lights`).
Downstream stages read upstream keys directly off the dict. There is no schema
class; the contract is implicit in the keys read/written.

When adding a new stage: insert it in the correct position in
`generate_scene`, write its outputs to a new key, and update any later stage
that should consume it.

## 2. "Generator" Subsystem Class Per Stage

Every stage is implemented by a class named `*Generator` or `*Selector` with
a stable shape:

- Constructor takes the `llm` callable plus any of `clip_model`,
  `clip_preprocess`, `clip_tokenizer`, `object_retriever`.
- A `generate_*` / `select_*` method takes `scene` + `additional_requirements`
  + `used_assets`, returns the data the orchestrator splices back into `scene`.
- Per-call mutable state (notably `self.used_assets`) is set on the instance
  by the caller right before invocation, not passed as an argument.

Examples: `FloorPlanGenerator` (`rooms.py:23`), `WallGenerator` (`walls.py`),
`DoorGenerator` (`doors.py`), `WindowGenerator` (`windows.py`),
`ObjectSelector` (`object_selector.py:32`), `FloorObjectGenerator`
(`floor_objects.py:24`), `WallObjectGenerator` (`wall_objects.py`),
`CeilingObjectGenerator` (`ceiling_objects.py`), `SmallObjectGenerator`
(`small_objects.py`).

New stages should follow the same constructor signature and `used_assets`
convention so they wire into `Holodeck.__init__` cleanly.

## 3. LLM as an Injected Callable

The OpenAI client is wrapped once in `Holodeck.__init__`
(`ai2holodeck/generation/holodeck.py:77`) into a `llm: str -> str` lambda and
passed into every generator's constructor. Generators never construct their
own client. To swap models or providers, change only that wrapper; all
downstream code is provider-agnostic.

Prompts are built via `langchain.PromptTemplate` against templates centralized
in `ai2holodeck/generation/prompts.py`. New prompts go in `prompts.py`; new
generators construct a `PromptTemplate(input_variables=[...], template=...)`
in their constructor.

## 4. CLIP + SBERT Dual-Encoder Retrieval

`ObjathorRetriever` (`ai2holodeck/generation/objaverse_retriever.py:18`) loads
two precomputed feature matrices (CLIP image features, SBERT text features)
for the union of objathor + THOR assets, then scores queries by
`clip_similarity + sbert_similarity` with a CLIP-similarity cutoff. Asset
selection across the codebase (objects, doors, windows, floor materials) goes
through this retriever rather than touching the asset database directly.
`compute_size_difference` (`objaverse_retriever.py:118`) demonstrates the
re-ranking pattern: keep the retriever's score, subtract a domain penalty,
re-sort.

## 5. Constraint-Based Placement: DFS Default, MILP Optional

Object placement (floor and wall) supports two interchangeable solvers behind
the same generator interface:

- DFS solver (`DFS_Solver_Floor` in `floor_objects.py`, `DFS_Solver_Wall` in
  `wall_objects.py`) — the default and recommended path.
- MILP solver via `cvxpy` / `gurobipy`, helpers in
  `ai2holodeck/generation/milp_utils.py`.

The choice is toggled by `use_milp` flagged through from CLI →
`Holodeck.generate_scene` → `FloorObjectGenerator.use_milp`
(`floor_objects.py:51`). README explicitly recommends DFS for better layouts.

## 6. Path & Config Constants Module

All filesystem paths, model names, and version strings live in
`ai2holodeck/constants.py`. Anything env-overridable uses
`os.environ.get(NAME, default)`. `Holodeck.__init__` calls
`confirm_paths_exist()` (`holodeck.py:38`) up front so missing data fails
loud before any LLM calls. Add new paths/versions here, not inline.

## 7. Retry-with-Reset at the Orchestrator Boundary

`generate_single_scene` in `ai2holodeck/main.py:51` wraps the entire
`generate_scene` call in a bare `try/except` retry loop (`max_retries=10`)
that re-runs from scratch on any non-`KeyboardInterrupt` exception. Internal
stages are therefore allowed to raise on failure — recovery is the
orchestrator's job, not each generator's. `KeyboardInterrupt` is explicitly
re-raised so Ctrl-C still works (`main.py:68`).

## 8. CLI Booleans as Strings + `ast.literal_eval`

CLI flags that are conceptually booleans (`--generate_image`, `--use_milp`,
`--add_ceiling`, `--single_room`, …) are declared as strings with defaults
like `"True"` / `"False"` and parsed with `ast.literal_eval` at the call site
(`main.py:60-66`). When adding a new boolean flag, follow this pattern rather
than `argparse`'s native `store_true`, because shell scripts in
`hd_mr*.sh` pass these as quoted strings.
