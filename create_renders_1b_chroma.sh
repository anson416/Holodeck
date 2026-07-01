#!/usr/bin/env bash
# Phase 1b (chromatic) -- chromatic background sweep (paper Table 1: red, green,
# blue). Composited over the baseline-camera master.
for scene_file in data/scenes/*/*.json; do
    [ -e "$scene_file" ] || continue
    echo "Processing: $scene_file"
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 512 \
        --bg-color "255,0,0;0,255,0;0,0,255" \
        --hdri city \
        --focal 50 \
        --pitches 90 \
        --yaws 0 \
        --fit-ratio 1.0 \
        --no-cull-walls
done
echo "Batch rendering complete."
