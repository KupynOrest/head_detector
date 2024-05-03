from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fire import Fire
from infery.utils.importing.lazy_imports import cv2
from super_gradients.training import models
import scipy.io
from super_gradients.training.utils.utils import infer_model_device

from yolo_head.dataset_parsing import draw_2d_keypoints
from yolo_head.flame import FlameParams, FLAME_CONSTS
from yolo_head.yolo_heads_predictions import YoloHeadsPredictions


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


def predict_on_image(model, image, dsize=320):
    device = infer_model_device(model)

    # Resize image to dsize x dsize but keep the aspect ratio by padding with zeros
    original_shape = image.shape[:2]
    scale_w, scale_h = dsize / original_shape[1], dsize / original_shape[0]
    if scale_w > scale_h:
        new_shape = (dsize, int(original_shape[0] * scale_w))
        scale = scale_w
    else:
        new_shape = (int(original_shape[1] * scale_h), dsize)
        scale = scale_h

    image = cv2.resize(image, new_shape)

    # Pad the image with zeros
    # For simplicity, we do bottom and right padding to simply the calculations in post-processing
    pad_w = dsize - new_shape[1]
    pad_h = dsize - new_shape[0]
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=127)

    image_input = torch.from_numpy(image).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    raw_predictions = model(image_input)
    (predictions,) = model.get_post_prediction_callback(conf=0.1, iou=0.5)(raw_predictions)

    predictions: YoloHeadsPredictions = predictions
    predictions.predicted_2d_vertices /= scale # There are 565 keypoints subset here
    predictions.predicted_3d_vertices /= scale # There are 565 keypoints subset here

    flame = FlameParams.from_3dmm(predictions.mm_params, FLAME_CONSTS)
    flame.scale /= scale

    predictions.mm_params = flame.to_3dmm_tensor()

    return predictions


def main(model_name="YoloHeads_M", checkpoint="C:/Develop/GitHub/VGG/head_detector/yolo_head_training/weights/ckpt_best.pth", dataset_dir="g:/AFLW2000"):
    images, labels = find_images_and_labels(dataset_dir)
    model = models.get(model_name, checkpoint_path=checkpoint, num_classes=413).eval() # 412 is total number of flame params

    for image_path, gt in zip(images[:10], labels[:10]):
        image = cv2.imread(str(image_path))
        # image = cv2.resize(image, (640, 640))
        predictions = predict_on_image(model, image)

        plt.figure()
        plt.imshow(draw_2d_keypoints(image[..., ::-1], predictions.predicted_2d_vertices.reshape(-1, 2))) # 565
        plt.show()

        # Load the ground truth
        gt = scipy.io.loadmat(gt)
        gt_pose = np.array(gt["Pose_Para"])  # (1, 7) shape ??? 3 rotation, 3 translation, 1 scale ?
        gt_pts3d = np.array(gt["pts3d_68"])  # (3, 68) shape


if __name__ == "__main__":
    Fire(main)
