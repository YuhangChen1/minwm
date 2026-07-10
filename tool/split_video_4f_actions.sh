#!/usr/bin/env bash
set -euo pipefail

read -r -p "Input video path: " input_video
read -r -p "Target folder: " target_dir

if [[ -z "$input_video" || ! -f "$input_video" ]]; then
  echo "Input video does not exist: $input_video" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffmpeg and ffprobe are required." >&2
  exit 1
fi

mkdir -p "$target_dir/actions"

fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$input_video")"
if [[ -z "$fps" || "$fps" == "0/0" ]]; then
  fps="24/1"
fi

num_frames="$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nw=1:nk=1 "$input_video")"
if [[ -z "$num_frames" || "$num_frames" == "N/A" ]]; then
  num_frames="$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$input_video")"
fi

if ! [[ "$num_frames" =~ ^[0-9]+$ ]] || [[ "$num_frames" -lt 1 ]]; then
  echo "Could not determine frame count for: $input_video" >&2
  exit 1
fi

step_frames=4

ffmpeg -hide_banner -loglevel error -y \
  -i "$input_video" \
  -vf "select='eq(n,0)'" \
  -vsync 0 \
  "$target_dir/first_frame.png"

manifest="$target_dir/manifest.json"
{
  printf '{\n'
  printf '  "source_video": "%s",\n' "$input_video"
  printf '  "fps": "%s",\n' "$fps"
  printf '  "source_num_frames": %s,\n' "$num_frames"
  printf '  "step_frames": %s,\n' "$step_frames"
  printf '  "initial_frame": {\n'
  printf '    "frame_index": 0,\n'
  printf '    "file": "first_frame.png"\n'
  printf '  },\n'
  printf '  "actions": [\n'
} > "$manifest"

action_index=0
start_frame=1
first_entry=1

while [[ "$start_frame" -lt "$num_frames" ]]; do
  end_frame=$((start_frame + step_frames - 1))
  if [[ "$end_frame" -ge "$num_frames" ]]; then
    end_frame=$((num_frames - 1))
  fi

  frame_count=$((end_frame - start_frame + 1))
  action_name="$(printf 'action_%02d_frames_%d_%d.mp4' "$action_index" "$start_frame" "$end_frame")"
  output_video="$target_dir/actions/$action_name"

  ffmpeg -hide_banner -loglevel error -y \
    -i "$input_video" \
    -vf "select='between(n,${start_frame},${end_frame})',setpts=N/(${fps})/TB" \
    -an -r "$fps" -frames:v "$frame_count" \
    -c:v libx264 -pix_fmt yuv420p \
    "$output_video"

  if [[ "$first_entry" -eq 0 ]]; then
    printf ',\n' >> "$manifest"
  fi
  first_entry=0

  printf '    {' >> "$manifest"
  printf '"action_index": %s, ' "$action_index" >> "$manifest"
  printf '"action": "action_%02d", ' "$action_index" >> "$manifest"
  printf '"source_frame_start": %s, ' "$start_frame" >> "$manifest"
  printf '"source_frame_end": %s, ' "$end_frame" >> "$manifest"
  printf '"num_frames": %s, ' "$frame_count" >> "$manifest"
  printf '"file": "actions/%s"' "$action_name" >> "$manifest"
  printf '}' >> "$manifest"

  action_index=$((action_index + 1))
  start_frame=$((end_frame + 1))
done

{
  printf '\n'
  printf '  ]\n'
  printf '}\n'
} >> "$manifest"

echo "Done."
echo "Initial frame: $target_dir/first_frame.png"
echo "Action clips:  $target_dir/actions"
echo "Manifest:      $manifest"
echo "Total clips:   $action_index"
