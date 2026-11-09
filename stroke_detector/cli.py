"""Command-line interface for running the stroke detection pipeline on an image."""

import argparse
import json
from pathlib import Path
import sys
import time
import cv2

from .config import Config
from .pipeline import run_pipeline
from .viz import save_stage_visualizations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: python -m stroke_detector.cli input.jpg --out out/ --debug."""
    parser = argparse.ArgumentParser(
        description="Detect thin hand-drawn strokes on textured cardboard and export vector/SVG data."
    )
    parser.add_argument("input", type=str, help="Path to input JPEG/PNG image")
    parser.add_argument(
        "--out",
        type=str,
        default="out",
        help="Output directory for JSON, SVG, and debug visualizations",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save 01_capture ... 09_output side-by-side visualization PNGs",
    )
    parser.add_argument(
        "--aruco",
        action="store_true",
        help="Enable ArUco 4-corner detection and perspective flattening prior to Stage 1",
    )
    parser.add_argument(
        "--auto-roi",
        action="store_true",
        help="Auto-detect cardboard workpiece ROI, zoom into it for Stages 1-6, and scale coordinates back to full bed",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Explicit ROI bounding box 'x,y,w,h' in bed pixels to zoom into before Stage 2",
    )
    parser.add_argument(
        "--ridge",
        type=str,
        default="sato",
        choices=["sato", "frangi", "meijering", "blackhat"],
        help="Ridge enhancement filter type for Stage 3",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file not found: {in_path}", file=sys.stderr)
        return 1

    img_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Error: Failed to decode image: {in_path}", file=sys.stderr)
        return 1

    roi_bbox = None
    if args.roi:
        parts = [int(p.strip()) for p in args.roi.split(",")]
        if len(parts) == 4:
            roi_bbox = (parts[0], parts[1], parts[2], parts[3])

    cfg = Config(
        enable_aruco_warp=args.aruco,
        auto_detect_roi=args.auto_roi,
        roi_bbox=roi_bbox,
        ridge_filter=args.ridge,
    )

    t0 = time.perf_counter()
    result_dict, artifacts = run_pipeline(img_bgr, cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "output.svg"
    svg_path.write_text(str(result_dict.pop("svg", "")), encoding="utf-8")

    json_path = out_dir / "result.json"
    result_dict["runtime_ms"] = round(elapsed_ms, 2)
    json_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")

    if args.debug:
        save_stage_visualizations(artifacts, out_dir)

    print(json.dumps(result_dict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

