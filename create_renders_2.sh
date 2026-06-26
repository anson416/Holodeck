for scene_file in data/scenes/*/*.json; do
    # Check if files exist to avoid running on an empty glob
    [ -e "$scene_file" ] || continue

    echo "Processing: $scene_file"

    # Execute the python command
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 512 \
        --bg-color 128,128,128 \
        --hdri city \
        --focal 50 \
        --pitches 90,60 \
        --yaws 0,30,60,90,120,150,180,210,240,270,300,330 \
        --fit-ratio 1.0 \
        --no-cull-walls
done

echo "Batch rendering complete."
