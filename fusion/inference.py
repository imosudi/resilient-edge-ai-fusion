"""
Inference deployment profiles for CPU, GPU, and Hailo NPU execution.

The project has three benchmark profiles:
- CPU baseline: ONNX model, FP32 precision, ONNX Runtime.
- GPU comparison: PyTorch/Ultralytics model, FP32 precision, CUDA.
- NPU accelerated: HEF model, INT8 precision, Hailo Runtime.

This module keeps those definitions in one place so pipeline outputs,
metrics, and CLI options all describe inference consistently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


CPU_TARGET = "cpu"
GPU_TARGET = "gpu"
NPU_TARGET = "npu"
INFERENCE_TARGET_CHOICES = (CPU_TARGET, GPU_TARGET, NPU_TARGET)


@dataclass(frozen=True)
class InferenceProfile:
    target: str
    label: str
    model_artifact: str
    precision: str
    runtime: str
    accelerator: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


CPU_INFERENCE_PROFILE = InferenceProfile(
    target=CPU_TARGET,
    label="CPU FP32 ONNX baseline",
    model_artifact="onnx",
    precision="FP32",
    runtime="ONNX Runtime",
    accelerator="Raspberry Pi 5 CPU",
)

GPU_INFERENCE_PROFILE = InferenceProfile(
    target=GPU_TARGET,
    label="GPU FP32 CUDA comparison",
    model_artifact="pt",
    precision="FP32",
    runtime="PyTorch CUDA",
    accelerator="CUDA GPU",
)

NPU_INFERENCE_PROFILE = InferenceProfile(
    target=NPU_TARGET,
    label="Hailo NPU INT8 HEF accelerated",
    model_artifact="hef",
    precision="INT8",
    runtime="Hailo Runtime",
    accelerator="Hailo-8L",
)


def get_inference_profile(target: str) -> InferenceProfile:
    normalised_target = target.lower().strip()

    if normalised_target == CPU_TARGET:
        return CPU_INFERENCE_PROFILE

    if normalised_target in {GPU_TARGET, "cuda"}:
        return GPU_INFERENCE_PROFILE

    if normalised_target in {NPU_TARGET, "hailo"}:
        return NPU_INFERENCE_PROFILE

    raise ValueError(
        f"Unsupported inference target: {target}. "
        f"Expected one of: {', '.join(INFERENCE_TARGET_CHOICES)}"
    )
