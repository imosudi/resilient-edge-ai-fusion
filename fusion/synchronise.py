def synchronise(camera_frames, lidar_scans):

    """
    Match camera frames with nearest LiDAR timestamp.
    """

    synchronised = []

    for frame in camera_frames:

        frame_ts = frame["timestamp"]

        nearest_scan = min(
            lidar_scans,
            key=lambda scan: abs(
                scan["timestamp"] - frame_ts
            )
        )

        synchronised.append({
            "frame": frame,
            "lidar": nearest_scan
        })

    return synchronised


if __name__ == "__main__":

    camera_frames = [
        {
            "timestamp": 1.002,
            "image": "frame_001.jpg"
        },
        {
            "timestamp": 2.005,
            "image": "frame_002.jpg"
        }
    ]

    lidar_scans = [
        {
            "timestamp": 1.010,
            "ranges": [1000, 1020, 980]
        },
        {
            "timestamp": 2.001,
            "ranges": [990, 1015, 975]
        }
    ]

    synced = synchronise(
        camera_frames,
        lidar_scans
    )

    print(synced)