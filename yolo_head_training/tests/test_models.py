import torch
from super_gradients.training.utils import HpmStruct

from yolo_head import YoloHeads_M, YoloHeads, YoloHeadsPostPredictionCallback
from yolo_head.flame import FLAMELayer, FLAME_CONSTS, FlameParams, \
    get_445_keypoints_indexes, reproject_spatial_vertices


def test_flame():
    flame = FLAMELayer(consts=FLAME_CONSTS)
    params = torch.randn(1, 300 + 100 + 6 + 3 + 1 + 3)

    predicted_3d_vertices = reproject_spatial_vertices(flame, params, to_2d=False)
    predicted_2d_vertices = predicted_3d_vertices[..., :2]
    print(predicted_2d_vertices.size())


def test_flame_reproject():
    flame = FLAMELayer(consts=FLAME_CONSTS)
    num_flame_params = 300 + 100 + 6 + 3 + 1 + 3
    params = torch.zeros((0, num_flame_params))
    x = reproject_spatial_vertices(flame, params)
    print(x.size())

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
