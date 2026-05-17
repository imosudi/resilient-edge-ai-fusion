import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.synchronise import synchronise


def test_basic_synchronisation():

    camera_frames = [
        {"timestamp": 1.0, "image": "frame1"},
        {"timestamp": 2.0, "image": "frame2"}
    ]

    lidar_scans = [
        {"timestamp": 1.1, "ranges": []},
        {"timestamp": 2.1, "ranges": []}
    ]

    synced = synchronise(
        camera_frames,
        lidar_scans
    )

    assert len(synced) == 2

    assert synced[0]["lidar"]["timestamp"] == 1.1
    assert synced[1]["lidar"]["timestamp"] == 2.1