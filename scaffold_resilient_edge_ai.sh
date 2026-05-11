#!/usr/bin/env bash
# scaffold_resilient_edge_ai.sh
# Creates the full directory and file structure for the resilient_edge_ai project.
# Usage:  bash scaffold_resilient_edge_ai.sh [target_directory]
# If no target_directory is given, the project is created in the current directory.

set -euo pipefail

ROOT="${1:-resilient_edge_ai}"

echo "Scaffolding project at: $(realpath "$ROOT")"


dirs=(
    configs

    dataset/images
    dataset/labels
    dataset/lidar
    dataset/degraded
    dataset/synchronised

    preprocessing/vision
    preprocessing/lidar
    preprocessing/synchronisation

    models/yolov8
    models/exported
    models/hailo

    fusion
    metrics
    notebooks
    scripts
    logs
    exports
    tests
)

for d in "${dirs[@]}"; do
    mkdir -p "$ROOT/$d"
    # Keep empty directories tracked in git
    touch "$ROOT/$d/.gitkeep"
done


cat > "$ROOT/configs/dataset_config.yaml" << 'EOF'
# dataset_config.yaml
# Paths and parameters for dataset ingestion and splits.

dataset:
  root: dataset/
  images_dir: dataset/images/
  labels_dir: dataset/labels/
  lidar_dir: dataset/lidar/
  degraded_dir: dataset/degraded/
  synchronised_dir: dataset/synchronised/

splits:
  train: 0.70
  val: 0.15
  test: 0.15
  seed: 42

classes:
  - pedestrian
  - cyclist
  - vehicle

degradation:
  enabled: true
  types:
    - fog
    - rain
    - noise
EOF

cat > "$ROOT/configs/model_config.yaml" << 'EOF'
# model_config.yaml
# YOLOv8 architecture and training hyperparameters.

model:
  architecture: yolov8n        # n / s / m / l / x
  pretrained_weights: yolov8n.pt
  input_size: [640, 640]
  num_classes: 3

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  optimizer: AdamW
  early_stopping_patience: 15
  device: cpu                  # cpu | cuda | hailo

export:
  formats:
    - onnx
    - hef                      # Hailo Execution Format
  output_dir: models/exported/
  hailo_output_dir: models/hailo/
EOF

cat > "$ROOT/configs/fusion_config.yaml" << 'EOF'
# fusion_config.yaml
# Late-fusion strategy for vision + LiDAR modalities.

fusion:
  strategy: weighted_average   # weighted_average | nms | learned
  modalities:
    - name: vision
      weight: 0.6
    - name: lidar
      weight: 0.4

  nms:
    iou_threshold: 0.45
    confidence_threshold: 0.25

  fallback:
    # Which modality to use when one stream is unavailable
    vision_unavailable: lidar
    lidar_unavailable: vision
EOF


cat > "$ROOT/requirements.txt" << 'EOF'
# Core
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
onnx>=1.14.0
onnxruntime>=1.16.0

# Data / preprocessing
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
open3d>=0.17.0            # LiDAR point-cloud processing
scipy>=1.11.0
pyyaml>=6.0

# Metrics / visualisation
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pycocotools>=2.0.6

# Notebooks
jupyter>=1.0.0
ipywidgets>=8.0.0

# Utilities
tqdm>=4.65.0
loguru>=0.7.0
python-dotenv>=1.0.0
EOF

# ─── README ──────────────────────────────────────────────────────────────────

cat > "$ROOT/README.md" << 'EOF'
# resilient_edge_ai

Multi-modal (vision + LiDAR) object detection pipeline targeting edge deployment
on Hailo AI HAT+ with robustness to sensor degradation (fog, rain, noise).

## Project Layout

```
resilient_edge_ai/
├── configs/               YAML configuration for dataset, model, and fusion
├── dataset/               Raw and processed data
│   ├── images/            RGB camera frames
│   ├── labels/            YOLO-format annotation .txt files
│   ├── lidar/             Point-cloud files (.pcd / .bin)
│   ├── degraded/          Synthetically degraded copies
│   └── synchronised/      Temporally aligned vision-LiDAR pairs
├── preprocessing/         Modality-specific preprocessing modules
│   ├── vision/
│   ├── lidar/
│   └── synchronisation/
├── models/
│   ├── yolov8/            Training artefacts and checkpoints
│   ├── exported/          ONNX and other intermediate exports
│   └── hailo/             Compiled .hef files for Hailo runtime
├── fusion/                Late-fusion logic
├── metrics/               mAP, latency, and resilience evaluation
├── notebooks/             Exploratory and reporting notebooks
├── scripts/               CLI entry points (train, export, evaluate, deploy)
├── logs/                  Training and inference logs
├── exports/               Final deliverables (reports, plots)
└── tests/                 Unit and integration tests
```

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure paths
cp configs/dataset_config.yaml configs/dataset_config.local.yaml
# Edit the local copy to point at your data

# 4. Train
python scripts/train.py --config configs/model_config.yaml

# 5. Export to Hailo
python scripts/export.py --format hef --config configs/model_config.yaml
```

## Configuration

All runtime behaviour is controlled through YAML files in `configs/`:

| File | Purpose |
|---|---|
| `dataset_config.yaml` | Data paths, splits, class names, degradation types |
| `model_config.yaml` | Architecture, hyperparameters, export targets |
| `fusion_config.yaml` | Modality weights, NMS settings, fallback strategy |

## Hardware Target

Raspberry Pi 5 + Hailo AI HAT+ (26 TOPS).
Benchmarking scripts compare CPU / GPU / NPU inference latency using YOLOv8n
on MS COCO 2017 and the project-specific dataset.
EOF

# ─── PLACEHOLDER PYTHON MODULES ──────────────────────────────────────────────

# preprocessing stubs
for mod in vision lidar synchronisation; do
    cat > "$ROOT/preprocessing/$mod/__init__.py" << EOF
# preprocessing/$mod/__init__.py
EOF
done

# fusion / metrics stubs
for pkg in fusion metrics; do
    cat > "$ROOT/$pkg/__init__.py" << EOF
# $pkg/__init__.py
EOF
done

# tests stub
cat > "$ROOT/tests/__init__.py" << 'EOF'
# tests/__init__.py
EOF

# ─── .gitignore ──────────────────────────────────────────────────────────────

cat > "$ROOT/.gitignore" << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
*.egg-info/

# Data (large files — manage with DVC or Git LFS)
dataset/images/*
dataset/labels/*
dataset/lidar/*
dataset/degraded/*
dataset/synchronised/*
!dataset/**/.gitkeep

# Model weights
*.pt
*.onnx
*.hef

# Logs and outputs
logs/*
exports/*
!logs/.gitkeep
!exports/.gitkeep

# Jupyter
.ipynb_checkpoints/

# Local config overrides
configs/*.local.yaml

# Environment
.env
EOF


echo ""
echo "Done. Structure created:"
find "$ROOT" | sort | sed "s|$ROOT/||" | sed "s|[^/]*/|  |g"
