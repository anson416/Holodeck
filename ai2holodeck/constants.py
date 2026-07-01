import os
from pathlib import Path

ABS_PATH_OF_HOLODECK = os.path.abspath(os.path.dirname(Path(__file__)))

ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")
HD_BASE_VERSION = os.environ.get("HD_BASE_VERSION", "2023_09_23")

OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser("~/.objathor-assets")
)

# Use VLMUNR_OBJATHOR_ROOT to point at the server's objathor store when the
# local Mac paths don't exist (falls back to the env-var base dir).
_VLMUNR_OBJATHOR_ROOT = os.environ.get(
    "VLMUNR_OBJATHOR_ROOT",
    "/research/d2/fyp24/yflam1/objathor-assets/2023_09_23"
    if os.path.isdir("/research/d2/fyp24/yflam1/objathor-assets/2023_09_23")
    else None,
)
OBJATHOR_ASSETS_DIR = (
    _VLMUNR_OBJATHOR_ROOT + "/assets"
    if _VLMUNR_OBJATHOR_ROOT
    else os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION, "assets")
)
OBJATHOR_FEATURES_DIR = (
    _VLMUNR_OBJATHOR_ROOT + "/features"
    if _VLMUNR_OBJATHOR_ROOT
    else os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION, "features")
)
OBJATHOR_ANNOTATIONS_PATH = (
    _VLMUNR_OBJATHOR_ROOT + "/annotations.json.gz"
    if _VLMUNR_OBJATHOR_ROOT
    else os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION, "annotations.json.gz")
)

HOLODECK_BASE_DATA_DIR = os.path.join(
    OBJATHOR_ASSETS_BASE_DIR, "holodeck", HD_BASE_VERSION
)

HOLODECK_THOR_FEATURES_DIR = os.path.join(
    HOLODECK_BASE_DATA_DIR, "thor_object_data"
)
HOLODECK_THOR_ANNOTATIONS_PATH = os.path.join(
    HOLODECK_BASE_DATA_DIR, "thor_object_data", "annotations.json.gz"
)

if ASSETS_VERSION > "2023_09_23":
    THOR_COMMIT_ID = "8524eadda94df0ab2dbb2ef5a577e4d37c712897"
else:
    THOR_COMMIT_ID = "3213d486cd09bcbafce33561997355983bdf8d1a"

# LLM_MODEL_NAME = "gpt-4-1106-preview"
# Text-layout role standardised on a single pinned snapshot for the audit
# (Holodeck's high-level layout reasoning is text-agnostic). Overridable via
# --model_name on the CLI.
LLM_MODEL_NAME = "gpt-5.1-2025-11-13"

DEBUGGING = os.environ.get("DEBUGGING", "0").lower() in [
    "1",
    "true",
    "True",
    "t",
    "T",
]
