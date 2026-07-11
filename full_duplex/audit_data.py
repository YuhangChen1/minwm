from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import decord

from full_duplex.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="full_duplex/configs/overfit.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("full_duplex/outputs/smallest_000000/data_audit.json"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["project_root"])
    semantic_manifest_path = Path(config["action_manifest"])
    script_manifest_path = root / "dataset/SmallestData/script_test_split/manifest.json"
    semantic = json.loads(semantic_manifest_path.read_text(encoding="utf-8"))
    script = json.loads(script_manifest_path.read_text(encoding="utf-8"))
    video = decord.VideoReader(config["video_path"], num_threads=1)

    comparisons = []
    for semantic_action, script_action in zip(semantic["actions"], script["actions"], strict=True):
        semantic_clip = semantic_manifest_path.parent / semantic_action["file"]
        script_clip = script_manifest_path.parent / script_action["file"]
        semantic_hash = _sha256(semantic_clip)
        script_hash = _sha256(script_clip)
        semantic_frames = semantic_action["source_frames"]
        script_frames = list(
            range(script_action["source_frame_start"], script_action["source_frame_end"] + 1)
        )
        comparison = {
            "turn": semantic_action["action_index"],
            "semantic_label": semantic_action["action"],
            "source_frames": semantic_frames,
            "semantic_clip": str(semantic_clip.resolve()),
            "provided_script_clip": str(script_clip.resolve()),
            "sha256": semantic_hash,
            "bitwise_identical": semantic_hash == script_hash,
            "frame_ranges_identical": semantic_frames == script_frames,
            "semantic_clip_frames": len(decord.VideoReader(str(semantic_clip), num_threads=1)),
            "script_clip_frames": len(decord.VideoReader(str(script_clip), num_threads=1)),
        }
        comparisons.append(comparison)
    if len(comparisons) != 19 or not all(
        row["bitwise_identical"] and row["frame_ranges_identical"] for row in comparisons
    ):
        raise AssertionError("The two action split directories are not equivalent")

    external_path = Path(
        "/mnt/onelab0/sub5-v2u2/cyh_area/data/0data/minWM/dataset/SmallestData/"
        "split_4f_actions/actions"
    )
    report = {
        "source_video": str(Path(config["video_path"]).resolve()),
        "source_video_frames": len(video),
        "source_video_fps": float(video.get_avg_fps()),
        "source_video_resolution": [video[0].shape[0], video[0].shape[1]],
        "semantic_manifest": str(semantic_manifest_path.resolve()),
        "provided_script_manifest": str(script_manifest_path.resolve()),
        "num_actions": len(comparisons),
        "semantic_labels": [row["semantic_label"] for row in comparisons],
        "all_clip_pairs_bitwise_identical": True,
        "external_mnt_path": str(external_path),
        "external_mnt_path_exists": external_path.exists(),
        "path_resolution": (
            "The prompt's script_test_split clips exist and are bitwise identical to the "
            "semantic split_4f_actions clips. The semantic manifest is used because the "
            "script manifest names actions action_00..action_18 and has no right/a labels."
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
