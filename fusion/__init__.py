__all__ = [
    "FusionPipeline",
    "CameraCapture",
    "load_image",
    "LidarCapture",
    "polar_to_cartesian",
    "synchronise",
    "adaptive_fusion",
]


def __getattr__(name):
    if name == "FusionPipeline":
        from .pipeline import FusionPipeline

        return FusionPipeline

    if name in {"CameraCapture", "load_image"}:
        from .vision import CameraCapture, load_image

        return {"CameraCapture": CameraCapture, "load_image": load_image}[name]

    if name in {"LidarCapture", "polar_to_cartesian"}:
        from .lidar import LidarCapture, polar_to_cartesian

        return {
            "LidarCapture": LidarCapture,
            "polar_to_cartesian": polar_to_cartesian,
        }[name]

    if name == "synchronise":
        from .synchronise import synchronise

        return synchronise

    if name == "adaptive_fusion":
        from .confidence_fusion import adaptive_fusion

        return adaptive_fusion

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
