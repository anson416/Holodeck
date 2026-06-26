#!/usr/bin/env bash
# Phase 2 (pitch) -- camera pitch sweep at the baseline azimuth (yaw 0).
# Camera convention: pitch 90 == top-down, pitch 0 == eye-level. The paper's
# 7 pitch levels {0,15,30,45,60,75,90} are swept as tilt-from-top-down.
for scene_file in data/scenes/*/*.json; do
    [ -e "$scene_file" ] || continue
    echo "Processing: $scene_file"
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 512 \
        --bg-color 128,128,128 \
        --hdri city \
        --focal 50 \
        --pitches 90,75,60,45,30,15,0 \
        --yaws 0 \
        --fit-ratio 1.0 \
        --no-cull-walls
done
echo "Batch rendering complete."
