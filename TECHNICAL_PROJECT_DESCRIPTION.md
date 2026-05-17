# resilient-edge-ai-fusion — Technical Project Description

## Project Overview

`resilient-edge-ai-fusion` is a failure-aware multimodal edge AI framework designed to investigate resilient perception under degraded sensing conditions using adaptive Vision–LiDAR fusion on resource-constrained embedded hardware.

The project targets real-time intelligent perception at the network edge using:

- Raspberry Pi 5
- Hailo AI HAT+ (Hailo-8L)
- Raspberry Pi Camera Module 3
- Hokuyo URG-04LX-UG-01 LiDAR

The system combines:

- semantic perception from computer vision
- geometric spatial awareness from LiDAR
- adaptive confidence-based fusion
- failure compensation logic
- robustness benchmarking
- hardware-accelerated inference

to maintain operational perception under adverse environmental conditions and partial sensor failure.

The architecture is explicitly designed for:

- edge AI research
- resilient AIoT systems
- embedded multimodal inference
- failure-aware perception
- intelligent redundancy
- low-power AI deployment

within the computational and thermal constraints of embedded edge devices.

## Core Research Objective

The central research objective is to determine whether intelligent Vision–LiDAR fusion can improve resilience, operational continuity, and perceptual robustness in edge AI systems operating under degraded sensing conditions.

The project evaluates:

- how perception deteriorates under sensor degradation
- how adaptive fusion mitigates failure
- how fallback strategies preserve operational awareness
- how embedded AI accelerators affect real-time viability

while maintaining:

- low power consumption
- real-time performance
- deployable embedded footprints

## System Architecture

The architecture consists of six major subsystems.

### 1. Vision Perception Pipeline

**Purpose**

Provide semantic scene understanding through object detection.

**Components**

| Component | Function |
| --- | --- |
| Raspberry Pi Camera Module 3 | RGB image acquisition |
| YOLOv8n | Object detection |
| OpenCV | Image handling |
| Ultralytics | Inference engine |

**Outputs**

- bounding boxes
- class labels
- confidence scores
- annotated frames

### 2. LiDAR Perception Pipeline

**Purpose**

Provide geometric spatial awareness independent of lighting conditions.

**Components**

| Component | Function |
| --- | --- |
| Hokuyo URG-04LX-UG-01 | Range sensing |
| USB serial interface | Data acquisition |
| `polar_to_cartesian()` | Coordinate transformation |

**Outputs**

- range measurements
- Cartesian projections
- geometric occupancy representation

### 3. Sensor Synchronisation Layer

**Purpose**

Temporally align visual frames with LiDAR scans.

**Components**

| Component | Function |
| --- | --- |
| timestamp acquisition | temporal alignment |
| nearest-neighbour matching | frame pairing |

**Outputs**

- synchronised multimodal samples

### 4. Fusion Engine

**Purpose**

Combine semantic and geometric confidence into unified perception.

**Components**

| Component | Function |
| --- | --- |
| confidence fusion | weighted inference |
| adaptive fallback | degraded sensor compensation |
| modality weighting | dynamic trust adjustment |

**Outputs**

- fused confidence scores
- resilient detection decisions

### 5. Robustness Evaluation Framework

**Purpose**

Quantify resilience under degraded sensing conditions.

**Components**

| Component | Function |
| --- | --- |
| degradation generators | controlled perturbations |
| robustness scoring | resilience measurement |
| metrics logging | experiment recording |

**Degradation Types**

| Degradation | Purpose |
| --- | --- |
| low-light | illumination failure |
| motion blur | dynamic camera degradation |
| Gaussian noise | sensor corruption |
| occlusion | partial obstruction |
| LiDAR dropout | range failure simulation |

### 6. Deployment and Acceleration Layer

**Purpose**

Optimise inference for embedded edge deployment.

**Components**

| Component | Function |
| --- | --- |
| ONNX export | model portability |
| Hailo Dataflow Compiler | NPU optimisation |
| INT8 quantisation | acceleration |
| Hailo runtime | hardware execution |
