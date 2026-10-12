"""Several YOLO weights at once (detector.weights as a list): pooled detections, overlapping
boxes merged into one, one ByteTrack over the merged boxes."""
import numpy as np
import pytest
import torch
from ultralytics.engine.results import Boxes

from turretx.tracker import YoloBackend, merge_overlapping


def test_merge_identical_and_nearly_identical_boxes():
    out = merge_overlapping([((10, 10, 50, 40), 0.6), ((12, 11, 52, 41), 0.9)])
    assert len(out) == 1
    box, conf = out[0]
    assert box == (10, 10, 52, 41) and conf == 0.9  # union box, best confidence


def test_merge_box_inside_another():
    out = merge_overlapping([((0, 0, 100, 60), 0.5), ((20, 10, 50, 40), 0.8)])
    assert out == [((0, 0, 100, 60), 0.8)]


def test_separate_and_barely_touching_boxes_stay_separate():
    dets = [((0, 0, 40, 30), 0.9), ((200, 200, 240, 230), 0.8), ((38, 0, 78, 30), 0.7)]
    assert len(merge_overlapping(dets)) == 3  # side-by-side cars must not fuse


def test_merge_is_transitive_and_order_invariant():
    chain = [((0, 0, 40, 40), 0.5), ((10, 0, 50, 40), 0.6), ((20, 0, 60, 40), 0.7)]
    a = merge_overlapping(chain)
    b = merge_overlapping(chain[::-1])
    assert a == b == [((0, 0, 60, 40), 0.7)]


def test_merge_empty():
    assert merge_overlapping([]) == []


class _FakeModel:
    """Stands in for an ultralytics model: returns fixed boxes per call."""

    def __init__(self, per_frame, names=None):
        self.per_frame = list(per_frame)
        self.names = names or {0: "vehicle"}

    def predict(self, frame, **kwargs):
        rows = [r for r in self.per_frame.pop(0) if r[1] >= kwargs["conf"]]
        data = torch.tensor([[*b, c, 0] for b, c in rows], dtype=torch.float32).reshape(-1, 6)

        class R:
            boxes = Boxes(data, frame.shape[:2])
        return [R()]


def _backend(models, conf=0.2):
    b = YoloBackend.__new__(YoloBackend)
    b._init_tracking({"conf": conf, "imgsz": 640}, models=[(m, None) for m in models])
    return b


def test_per_model_confidence_drops_weak_boxes_of_one_model_only():
    frame = np.zeros((480, 640, 3), np.uint8)
    toy = [[((50, 50, 90, 80), 0.3)]] * 3                       # faint toy car: kept at 0.2
    strip = [[((437, 0, 478, 470), 0.3), ((200, 200, 320, 330), 0.95)]] * 3  # weak edge strip: dropped at 0.5
    b = _backend([_FakeModel(toy), _FakeModel(strip)], conf=[0.2, 0.5])
    for _ in range(3):
        out = b.update(frame)
    xs = sorted(round(box[0]) for _, box, _ in out)
    assert xs == [50, 200]


def test_conf_list_must_match_weights():
    with pytest.raises(ValueError):
        _backend([_FakeModel([[]]), _FakeModel([[]])], conf=[0.2])


def test_two_models_seeing_the_same_vehicle_give_one_track():
    frame = np.zeros((480, 640, 3), np.uint8)
    car = [[((100 + 5 * i, 100, 160 + 5 * i, 140), 0.8)] for i in range(4)]
    car_b = [[((102 + 5 * i, 98, 163 + 5 * i, 142), 0.7)] for i in range(4)]
    b = _backend([_FakeModel(car), _FakeModel(car_b)])
    ids = set()
    for _ in range(4):
        out = b.update(frame)
        assert len(out) == 1
        ids.add(out[0][0])
    assert len(ids) == 1  # one stable track id


def test_models_seeing_different_vehicles_give_separate_tracks():
    frame = np.zeros((480, 640, 3), np.uint8)
    toy = [[((50, 50, 90, 80), 0.8)]] * 3
    makeshift = [[((300, 200, 420, 330), 0.9)]] * 3
    b = _backend([_FakeModel(toy), _FakeModel(makeshift)])
    for _ in range(3):
        out = b.update(frame)
    assert len(out) == 2 and len({tid for tid, _, _ in out}) == 2


def test_no_detections_is_empty_not_an_error():
    frame = np.zeros((480, 640, 3), np.uint8)
    b = _backend([_FakeModel([[]] * 2)])
    assert b.update(frame) == [] and b.update(frame) == []


def test_weights_list_loads_every_model_and_keeps_person_excluded():
    b = YoloBackend({"weights": ["yolov8n.pt", "yolov8n.pt"], "conf": 0.2, "imgsz": 320,
                     "exclude_classes": ["person"]})
    assert len(b.models) == 2
    for model, classes in b.models:
        assert all(model.names[i] != "person" for i in classes)


@pytest.mark.parametrize("weights", ["yolov8n.pt", ["yolov8n.pt"]])
def test_single_weights_as_string_or_list(weights):
    b = YoloBackend({"weights": weights, "conf": 0.2, "imgsz": 320, "exclude_classes": ["person"]})
    assert len(b.models) == 1 and b.model is b.models[0][0]
