#!/usr/bin/env bash
# Phase 1c -- environment-map (HDRI) lighting sweep (paper Table 1, 8 maps).
for scene_file in data/scenes/*/*.json; do
    [ -e "$scene_file" ] || continue
    echo "Processing: $scene_file"
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 512 \
        --bg-color 128,128,128 \
        --hdri city,courtyard,forest,interior,night,studio,sunrise,sunset \
        --focal 50 \
        --pitches 90 \
        --yaws 0 \
        --fit-ratio 1.0 \
        --no-cull-walls
done
echo "Batch rendering complete."
