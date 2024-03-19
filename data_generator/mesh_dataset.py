import os
import json
from typing import Tuple, Union, Dict, Any

import cv2
import torch
import numpy as np
from fire import Fire
from tqdm import tqdm
from pycocotools.coco import COCO

from dad_3d_heads.predictor import FaceMeshPredictor


def extend_bbox(bbox: np.array, offset: Union[Tuple[float, ...], float] = 0.1) -> np.array:
    """
    Increases bbox dimensions by offset*100 percent on each side.

    IMPORTANT: Should be used with ensure_bbox_boundaries, as might return negative coordinates for x_new, y_new,
    as well as w_new, h_new that are greater than the image size the bbox is extracted from.

    :param bbox: [x, y, w, h]
    :param offset: (left, right, top, bottom), or (width_offset, height_offset), or just single offset that specifies
    fraction of spatial dimensions of bbox it is increased by.

    For example, if bbox is a square 100x100 pixels, and offset is 0.1, it means that the bbox will be increased by
    0.1*100 = 10 pixels on each side, yielding 120x120 bbox.

    :return: extended bbox, [x_new, y_new, w_new, h_new]
    """
    x, y, w, h = bbox

    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset

    return np.array([x - w * left, y - h * top, w * (1.0 + right + left), h * (1.0 + top + bottom)]).astype("int32")


def ensure_bbox_boundaries(bbox: np.array, img_shape: Tuple[int, int]) -> np.array:
    """
    Trims the bbox not the exceed the image.
    :param bbox: [x, y, w, h]
    :param img_shape: (h, w)
    :return: trimmed to the image shape bbox
    """
    x1, y1, w, h = bbox
    x1, y1 = min(max(0, x1), img_shape[1]), min(max(0, y1), img_shape[0])
    x2, y2 = min(max(0, x1 + w), img_shape[1]), min(max(0, y1 + h), img_shape[0])
    w, h = x2 - x1, y2 - y1
    return np.array([x1, y1, w, h]).astype("int32")


def process_result(result: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in result.items():
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            result[key] = value.squeeze().tolist()
    return result


def create_dataset(images_folder: str, coco_path: str, save_path: str):
    os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "annotations"), exist_ok=True)
    coco = COCO(coco_path)
    img_ids = coco.getImgIds()
    predictor = FaceMeshPredictor.dad_3dnet()
    for img_id in tqdm(img_ids):
        image_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(images_folder, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        bboxes = [x['bbox'] for x in anns]
        mesh_annotations = []
        for bbox in bboxes:
            x, y, w, h = ensure_bbox_boundaries(extend_bbox(np.array(bbox), 0.1), image.shape[:2])
            cropped_img = image[y: y + h, x: x + w]
            result = process_result(predictor(cropped_img))
            mesh_annotations.append({
                "bbox": bbox,
                "extended_bbox": [int(x), int(y), int(w), int(h)],
                **result
            })
        cv2.imwrite(os.path.join(save_path, "images", image_info['file_name']), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        with open(os.path.join(save_path, "annotations", image_info['file_name'].replace("jpg", "json")), "w") as f:
            json.dump(mesh_annotations, f)


if __name__ == '__main__':
    Fire(create_dataset)
