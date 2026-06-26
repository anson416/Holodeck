#!/usr/bin/env bash
# Phase 1a -- resolution sweep (paper Table 1). All other factors at baseline.
# Camera convention: pitch 90 == top-down in render_blender.py.
for scene_file in data/scenes/*/*.json; do
    [ -e "$scene_file" ] || continue
    echo "Processing: $scene_file"
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 196,224,256,336,384,448,512,768,1024 \
        --bg-color 128,128,128 \
        --hdri city \
        --focal 50 \
        --pitches 90 \
        --yaws 0 \
        --fit-ratio 1.0 \
        --no-cull-walls
done
echo "Batch rendering complete."
