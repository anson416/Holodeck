#!/usr/bin/env bash
# Phase 1b -- background grayscale sweep (paper Table 1). The background is
# composited over a single baseline-camera master, so all other factors stay
# at baseline.
for scene_file in data/scenes/*/*.json; do
    [ -e "$scene_file" ] || continue
    echo "Processing: $scene_file"
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 512 \
        --bg-color "0,0,0;65,65,65;128,128,128;186,186,186;204,204,204;255,255,255" \
        --hdri city \
        --focal 50 \
        --pitches 90 \
        --yaws 0 \
        --fit-ratio 1.0 \
        --no-cull-walls
done
echo "Batch rendering complete."
