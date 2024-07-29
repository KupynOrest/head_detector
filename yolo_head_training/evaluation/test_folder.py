import os
import glob
from pathlib import Path
from typing import Union

import tqdm
import numpy as np
import torch
from fire import Fire
import cv2
from super_gradients.training import models
from head_mesh import HeadMesh
from super_gradients.training.utils.utils import infer_model_device
import random
from yolo_head.flame import FlameParams, FLAME_CONSTS
from yolo_head.yolo_heads_predictions import YoloHeadsPredictions
from evaluation.draw_utils import draw_3d_landmarks, get_relative_path


POINT_COLOR = (255, 255, 255)
HEAD_INDICES = np.load(str(get_relative_path("../yolo_head/flame_indices/head_indices.npy", __file__)),
                          allow_pickle=True)[()]


def find_images_and_labels(dataset_dir):
    dataset_dir = Path(dataset_dir)

    images = list(dataset_dir.glob("*.jpg"))
    labels = list(sorted(dataset_dir.glob("*.mat")))

    if len(images) != len(labels):
        raise ValueError(f"Number of images and labels do not match. There are {len(images)} images and {len(labels)} labels.")

    images = []
    for label_path in labels:
        image_path = dataset_dir / (label_path.stem + ".jpg")
        if not image_path.exists():
            raise ValueError(f"Image {image_path} does not exist")
        images.append(image_path)
    return images, labels


def predict_on_image(model, image, dsize=640):
    device = infer_model_device(model)

    # Resize image to dsize x dsize but keep the aspect ratio by padding with zeros
    original_shape = image.shape[:2]
    #resize to dsize max side
    scale = dsize / max(original_shape)
    new_shape = (int(original_shape[1] * scale), int(original_shape[0] * scale))
    image = cv2.resize(image, new_shape)

    # Pad the image with zeros
    # For simplicity, we do bottom and right padding to simply the calculations in post-processing
    pad_w = dsize - image.shape[1]
    pad_h = dsize - image.shape[0]
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=127)

    image_input = torch.from_numpy(image).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    raw_predictions = model(image_input)
    (predictions,) = model.get_post_prediction_callback(conf=0.4, iou=0.5, post_nms_max_predictions=100)(raw_predictions)

    predictions: YoloHeadsPredictions = predictions
    predictions.predicted_2d_vertices /= scale # There are 565 keypoints subset here
    predictions.predicted_3d_vertices /= scale # There are 565 keypoints subset here

    flame = FlameParams.from_3dmm(predictions.mm_params, FLAME_CONSTS)
    flame.scale /= scale
    predictions.mm_params = flame.to_3dmm_tensor()
    return predictions, flame


def limit_angle(angle: Union[int, float], pi: Union[int, float] = 180.0) -> Union[int, float]:
    """
    Angle should be in degrees, not in radians.
    If you have an angle in radians - use the function radians_to_degrees.
    """
    if angle < -pi:
        k = -2 * (int(angle / pi) // 2)
        angle = angle + k * pi
    if angle > pi:
        k = 2 * ((int(angle / pi) + 1) // 2)
        angle = angle - k * pi

    return angle


def main(pattern: str, model_name="YoloHeads_M", checkpoint="C:/Develop/GitHub/VGG/head_detector/yolo_head_training/weights/ckpt_best.pth"):
    images = glob.glob(pattern)
    os.makedirs("test", exist_ok=True)
    model = models.get(model_name, checkpoint_path=checkpoint, num_classes=413).eval() # 412 is total number of flame params
    index = 0

    head_mesh = HeadMesh()
    triangles = head_mesh.flame.faces
    filtered_triangles = []
    for triangle in triangles:
        keep = True
        for v in triangle:
            if v not in HEAD_INDICES:
                keep = False
        if keep:
            filtered_triangles.append(triangle)
    images = random.sample(images, 4000)
    for image_path in tqdm.tqdm(images):
        image = cv2.imread(str(image_path))
        predictions, flame_params = predict_on_image(model, image)
        if predictions.bboxes_xyxy.size()[0] == 0:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for vertices_2d in predictions.predicted_2d_vertices:
            image = draw_3d_landmarks(vertices_2d.reshape(-1, 2), image, filtered_triangles)
        cv2.imwrite(f"test/{index}.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        index += 1


if __name__ == "__main__":
    Fire(main)
