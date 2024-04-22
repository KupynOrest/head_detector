import torch
from super_gradients.training.utils import HpmStruct

from yolo_head import YoloHeads_M, YoloHeads, YoloHeadsPostPredictionCallback
from yolo_head.flame import FLAMELayer, FLAME_CONSTS, FlameParams, reproject_vertices, \
    get_445_keypoints_indexes


def test_flame():
    flame = FLAMELayer(consts=FLAME_CONSTS)
    params = torch.randn(1, 300 + 100 + 6 + 3 + 1 + 3)
    flame_params = FlameParams.from_3dmm(params, FLAME_CONSTS)

    predicted_3d_vertices = reproject_vertices(flame, flame_params, to_2d=False)
    predicted_2d_vertices = predicted_3d_vertices[..., :2]
    print(predicted_2d_vertices.size())


def test_get_445_keypoints():
    indexes = get_445_keypoints_indexes()
    print(indexes)

def test_model_forward():
    m = YoloHeads_M(arch_params=HpmStruct())
    outputs = m(torch.randn(1, 3, 640, 640))
    print(outputs)

    postprocess = YoloHeadsPostPredictionCallback(confidence_threshold=0.02, nms_iou_threshold=0.5, pre_nms_max_predictions=1000, post_nms_max_predictions=100)
    result = postprocess(outputs)
    assert True
