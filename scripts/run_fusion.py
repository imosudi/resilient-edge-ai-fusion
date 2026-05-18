# scripts/run_fusion.py
#
# Updated resilient Vision–LiDAR fusion runner
# - YOLOv8n object detection
# - Live annotation overlay
# - Confidence rendering
# - LiDAR projection visualisation
# - CPU / Hailo deployment awareness
#
# Requirements:
#   pip install ultralytics opencv-python numpy
#
# Example:
#   python scripts/run_fusion.py \
#       --mode live \
#       --display \
#       --yolo-model yolov8n.pt \
#       --inference-target cpu
#
# Notes:
# - Hailo inference integration remains abstracted inside FusionPipeline.
# - YOLO overlay is primarily for live visualisation and baseline validation.
#

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.hardware_config import (
    CAMERA_BACKEND_CHOICES,
    DEFAULT_CAMERA_BACKEND,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_HOKUYO_CLUSTER_COUNT,
    DEFAULT_HOKUYO_END_STEP,
    DEFAULT_HOKUYO_START_STEP,
    DEFAULT_LIDAR_BAUDRATE,
    DEFAULT_LIDAR_END_ANGLE_DEG,
    DEFAULT_LIDAR_MAX_DISTANCE_MM,
    DEFAULT_LIDAR_MIN_DISTANCE_MM,
    DEFAULT_LIDAR_PORT,
    DEFAULT_LIDAR_PROTOCOL,
    DEFAULT_LIDAR_START_ANGLE_DEG,
    DEFAULT_LIDAR_TIMEOUT,
    DEFAULT_MAX_FUSION_SAMPLES,
    DEFAULT_STREAM_DURATION,
    LIDAR_PROTOCOL_CHOICES,
)

from fusion.inference import INFERENCE_TARGET_CHOICES

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("scripts.run_fusion")


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run resilient Vision-LiDAR fusion pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["live", "offline"],
        default="offline",
    )

    parser.add_argument(
        "--image-folder",
        default="dataset/images",
    )

    parser.add_argument(
        "--lidar-log",
        default=None,
    )

    parser.add_argument(
        "--lidar-port",
        default=DEFAULT_LIDAR_PORT,
    )

    parser.add_argument(
        "--lidar-baudrate",
        type=int,
        default=DEFAULT_LIDAR_BAUDRATE,
    )

    parser.add_argument(
        "--lidar-timeout",
        type=float,
        default=DEFAULT_LIDAR_TIMEOUT,
    )

    parser.add_argument(
        "--lidar-protocol",
        choices=LIDAR_PROTOCOL_CHOICES,
        default=DEFAULT_LIDAR_PROTOCOL,
    )

    parser.add_argument(
        "--hokuyo-start-step",
        type=int,
        default=DEFAULT_HOKUYO_START_STEP,
    )

    parser.add_argument(
        "--hokuyo-end-step",
        type=int,
        default=DEFAULT_HOKUYO_END_STEP,
    )

    parser.add_argument(
        "--hokuyo-cluster-count",
        type=int,
        default=DEFAULT_HOKUYO_CLUSTER_COUNT,
    )

    parser.add_argument(
        "--lidar-start-angle",
        type=float,
        default=DEFAULT_LIDAR_START_ANGLE_DEG,
    )

    parser.add_argument(
        "--lidar-end-angle",
        type=float,
        default=DEFAULT_LIDAR_END_ANGLE_DEG,
    )

    parser.add_argument(
        "--lidar-min-distance",
        type=float,
        default=DEFAULT_LIDAR_MIN_DISTANCE_MM,
    )

    parser.add_argument(
        "--lidar-max-distance",
        type=float,
        default=DEFAULT_LIDAR_MAX_DISTANCE_MM,
    )

    parser.add_argument(
        "--camera-backend",
        choices=CAMERA_BACKEND_CHOICES,
        default=DEFAULT_CAMERA_BACKEND,
    )

    parser.add_argument(
        "--camera-width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
    )

    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_STREAM_DURATION,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_FUSION_SAMPLES,
    )

    parser.add_argument(
        "--display",
        action="store_true",
    )

    parser.add_argument(
        "--inference-target",
        choices=INFERENCE_TARGET_CHOICES,
        default="cpu",
    )

    # -----------------------------------------
    # YOLO
    # -----------------------------------------

    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLOv8 model path.",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="YOLO confidence threshold.",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Drawing Helpers
# ---------------------------------------------------------

def draw_detections(frame, detections):
    """
    Draw YOLO detections.
    """

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        confidence = det["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text = f"{label} {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    return frame


def draw_lidar_overlay(frame, lidar_points):
    """
    Overlay projected LiDAR points.
    """

    for point in lidar_points:

        px = int(point[0])
        py = int(point[1])

        cv2.circle(
            frame,
            (px, py),
            3,
            (255, 0, 0),
            -1,
        )

    return frame


def draw_metrics(frame, fps, inference_ms, target):
    """
    Draw runtime metrics.
    """

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Inference: {inference_ms:.1f} ms",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Target: {target.upper()}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    return frame


# ---------------------------------------------------------
# YOLO Inference
# ---------------------------------------------------------

def run_yolo(model, frame, conf_threshold=0.35):

    results = model.predict(
        frame,
        verbose=False,
        conf=conf_threshold,
    )

    detections = []

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls_id]

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

    return detections


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    from fusion.pipeline import FusionPipeline

    LOGGER.info("Loading YOLO model: %s", args.yolo_model)

    yolo_model = YOLO(args.yolo_model)

    if args.mode == "offline" and not args.lidar_log:
        raise SystemExit(
            "Offline mode requires --lidar-log."
        )

    pipeline = FusionPipeline(
        camera_source="folder" if args.mode == "offline" else "camera",
        image_folder=args.image_folder,
        camera_backend=args.camera_backend,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        lidar_port=args.lidar_port,
        lidar_baudrate=args.lidar_baudrate,
        lidar_timeout=args.lidar_timeout,
        lidar_log=args.lidar_log,
        lidar_protocol=args.lidar_protocol,
        hokuyo_start_step=args.hokuyo_start_step,
        hokuyo_end_step=args.hokuyo_end_step,
        hokuyo_cluster_count=args.hokuyo_cluster_count,
        lidar_start_angle=args.lidar_start_angle,
        lidar_end_angle=args.lidar_end_angle,
        lidar_min_distance=args.lidar_min_distance,
        lidar_max_distance=args.lidar_max_distance,
        inference_target=args.inference_target,
    )

    LOGGER.info("Starting fusion pipeline")

    results = pipeline.run_live(
        duration=args.duration,
        max_samples=args.max_samples,
        display=False,
    )

    frame_counter = 0
    start_time = time.time()

    for item in results:

        frame = item.get("frame")

        if frame is None:
            continue

        inference_start = time.time()

        detections = run_yolo(
            yolo_model,
            frame,
            conf_threshold=args.conf_threshold,
        )

        inference_ms = (time.time() - inference_start) * 1000

        lidar_points = item.get("projected_lidar", [])

        frame = draw_detections(frame, detections)
        frame = draw_lidar_overlay(frame, lidar_points)

        frame_counter += 1

        elapsed = time.time() - start_time
        fps = frame_counter / elapsed if elapsed > 0 else 0.0

        frame = draw_metrics(
            frame,
            fps,
            inference_ms,
            args.inference_target,
        )

        if args.display:

            cv2.imshow(
                "Resilient Edge AI Fusion",
                frame,
            )

            key = cv2.waitKey(1)

            if key == ord("q"):
                break

        LOGGER.info(
            "Frame=%d Detections=%d FPS=%.2f Inference=%.2fms",
            frame_counter,
            len(detections),
            fps,
            inference_ms,
        )

    cv2.destroyAllWindows()

    LOGGER.info("Fusion pipeline completed")


if __name__ == "__main__":
    main()