for scene_file in data/scenes/*/*.json; do
    # Check if files exist to avoid running on an empty glob
    [ -e "$scene_file" ] || continue

    echo "Processing: $scene_file"

    # Execute the python command
    python ai2holodeck/render_blender.py \
        --scene "$scene_file" \
        --resolutions 512 \
        --bg-color "0,0,0;18,18,18;65,65,65;117,117,117;128,128,128;186,186,186;204,204,204;255,255,255" \
        --hdri city \
        --focal 50 \
        --pitches 90 \
        --yaws 0 \
        --fit-ratio 1.0 \
        --no-cull-walls
done

echo "Batch rendering complete."
