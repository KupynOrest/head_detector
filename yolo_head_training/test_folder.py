import os
import math
import glob
from pathlib import Path
from typing import Union, Optional

from dataclasses import dataclass
import tqdm
import numpy as np
import torch
from fire import Fire
import cv2
from super_gradients.training import models
import scipy.io
from scipy.spatial.transform import Rotation
from super_gradients.training.utils.utils import infer_model_device

from yolo_head.dataset_parsing import draw_2d_keypoints
from yolo_head.flame import FlameParams, FLAME_CONSTS, rot_mat_from_6dof
from yolo_head.yolo_heads_predictions import YoloHeadsPredictions
from pytorch_toolbelt.utils import vstack_header


MAX_ROTATION = 99
MAE_THRESHOLD = 40

@dataclass
class RPY:
    roll: float
    pitch: float
    yaw: float


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
    (predictions,) = model.get_post_prediction_callback(conf=0.5, iou=0.5, post_nms_max_predictions=30)(raw_predictions)

    predictions: YoloHeadsPredictions = predictions
    predictions.predicted_2d_vertices /= scale # There are 565 keypoints subset here
    predictions.predicted_3d_vertices /= scale # There are 565 keypoints subset here

    flame = FlameParams.from_3dmm(predictions.mm_params, FLAME_CONSTS)
    flame.scale /= scale
    predictions.mm_params = flame.to_3dmm_tensor()
    params = predictions.mm_params.numpy().tolist()
    for i, head in enumerate(params):
        head = head[400:406]
        pred_pose = calculate_rpy(predictions.mm_params[i][400:406])
        print(f"Roll {pred_pose.roll:.2f}, Pitch {pred_pose.pitch:.2f}, Yaw {pred_pose.yaw:.2f}")
        print(["{0:0.2f}".format(i) for i in head])
    print("============================")

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


def calculate_rpy(flame_params) -> RPY:
    rot_mat = rot_mat_from_6dof(flame_params).numpy()[0]
    rot_mat_2 = np.transpose(rot_mat)
    angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
    roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
    return RPY(roll=roll, pitch=pitch, yaw=yaw)


def get_ground_truth(pose_path: str) -> Optional[RPY]:
    mat = scipy.io.loadmat(pose_path)
    pose_params = mat["Pose_Para"][0]
    degrees = pose_params[:3] * (180 / np.pi)
    if np.any(np.abs(degrees) > MAX_ROTATION):
        return None
    return RPY(roll=degrees[2], pitch=degrees[0], yaw=degrees[1])


def mae(x, y):
    PI = 180.0
    return min(
        math.fabs(x - y),
        math.fabs(x - (y - 2 * PI)),
        math.fabs(x - (y + 2 * PI)),
    )


def draw_pose(rpy: RPY, image: np.ndarray) -> np.ndarray:
    image = vstack_header(image, f"Roll: {rpy.roll:.2f}, Pitch: {rpy.pitch:.2f}, Yaw: {rpy.yaw:.2f}")
    return image


def main(folder: str, model_name="YoloHeads_M", checkpoint="C:/Develop/GitHub/VGG/head_detector/yolo_head_training/weights/ckpt_best.pth"):
    images = glob.glob(f"{folder}/*")
    os.makedirs("test", exist_ok=True)
    model = models.get(model_name, checkpoint_path=checkpoint, num_classes=413).eval() # 412 is total number of flame params
    index = 0
    for image_path in tqdm.tqdm(images):
        print(image_path)
        image = cv2.imread(str(image_path))
        predictions, flame_params = predict_on_image(model, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = draw_2d_keypoints(image[..., ::-1], predictions.predicted_2d_vertices.reshape(-1, 2))
        cv2.imwrite(f"test/{index}.jpg", image)
        index += 1


if __name__ == "__main__":
    Fire(main)