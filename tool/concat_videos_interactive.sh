#!/usr/bin/env bash
set -euo pipefail

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffmpeg and ffprobe are required." >&2
  exit 1
fi

read -r -p "Output video path: " output_video
if [[ -z "$output_video" ]]; then
  echo "Output video path is required." >&2
  exit 1
fi

echo "Enter input videos in order, one per line. Submit an empty line when done."

videos=()
while true; do
  read -r -p "Video $(( ${#videos[@]} + 1 )): " video_path
  if [[ -z "$video_path" ]]; then
    break
  fi
  if [[ ! -f "$video_path" ]]; then
    echo "File does not exist: $video_path" >&2
    exit 1
  fi
  videos+=("$video_path")
done

if [[ "${#videos[@]}" -lt 2 ]]; then
  echo "At least two input videos are required." >&2
  exit 1
fi

mkdir -p "$(dirname "$output_video")"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

list_file="$tmp_dir/concat_list.txt"
normalized_dir="$tmp_dir/normalized"
mkdir -p "$normalized_dir"

idx=0
for video_path in "${videos[@]}"; do
  normalized_file="$normalized_dir/$(printf '%04d' "$idx").mp4"
  ffmpeg -hide_banner -loglevel error -y \
    -i "$video_path" \
    -map 0:v:0 -an \
    -vf "setsar=1" \
    -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 18 \
    "$normalized_file"
  printf "file '%s'\n" "$normalized_file" >> "$list_file"
  idx=$((idx + 1))
done

ffmpeg -hide_banner -loglevel error -y \
  -f concat -safe 0 -i "$list_file" \
  -c copy \
  "$output_video"

echo "Done."
echo "Output video: $output_video"
echo "Input clips:  ${#videos[@]}"
