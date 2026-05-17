def adaptive_fusion(
    camera_conf,
    lidar_conf,
    camera_threshold=0.5
):

    """
    Adaptive confidence-based fusion.
    """

    # Camera degraded
    if camera_conf < camera_threshold:

        wc = 0.3
        wl = 0.7

    else:

        wc = 0.7
        wl = 0.3

    final_conf = (
        camera_conf * wc
        +
        lidar_conf * wl
    )

    return final_conf


if __name__ == "__main__":

    camera_conf = 0.42
    lidar_conf = 0.88

    fused_conf = adaptive_fusion(
        camera_conf,
        lidar_conf
    )

    print(f"Fused Confidence: {fused_conf:.4f}")



    