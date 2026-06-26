#!/usr/bin/env bash
# Phase 2 (yaw) -- camera azimuth sweep at a fixed oblique pitch (45 deg).
# 8 azimuths at 45-deg steps (paper Table 1). At a pure top-down view yaw is a
# trivial in-plane rotation, so the yaw sweep is run at the oblique pitch.
for scene_file in data/scenes/*/*.json; do
    [ -e "$scene_file" ] || continue
    echo "Processing: $scene_file"
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 512 \
        --bg-color 128,128,128 \
        --hdri city \
        --focal 50 \
        --pitches 45 \
        --yaws 0,45,90,135,180,225,270,315 \
        --fit-ratio 1.0 \
        --no-cull-walls
done
echo "Batch rendering complete."
