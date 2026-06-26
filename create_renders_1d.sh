#!/usr/bin/env bash
# Phase 1d -- focal-length sweep (paper Table 1). Other factors at baseline.
for scene_file in data/scenes/*/*.json; do
    [ -e "$scene_file" ] || continue
    echo "Processing: $scene_file"
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 512 \
        --bg-color 128,128,128 \
        --hdri city \
        --focal 16,24,35,50,85,100,200 \
        --pitches 90 \
        --yaws 0 \
        --fit-ratio 1.0 \
        --no-cull-walls
done
echo "Batch rendering complete."
