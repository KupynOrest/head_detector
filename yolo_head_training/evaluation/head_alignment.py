import os
from typing import Union, Tuple
import tqdm
import glob
import numpy as np
import torch
from fire import Fire
import cv2
from scipy.spatial.transform import Rotation
from super_gradients.training import models
from super_gradients.training.utils.utils import infer_model_device

from yolo_head.flame import FlameParams, FLAME_CONSTS, rot_mat_from_6dof, RPY
from evaluation.draw_utils import get_relative_path, HEAD_INDICES, draw_3d_landmarks


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


def extend_to_rect(bbox: np.array) -> np.array:
    x, y, w, h = bbox
    if w > h:
        diff = w - h
        return np.array([x, y - diff // 2, w, w])
    else:
        diff = h - w
        return np.array([x - diff // 2, y, h, h])


def vertically_align(
    img: np.ndarray, vertices: np.ndarray, flame_params: FlameParams, roll: float, scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    scull_center = flame_params_skull_center(flame_params, scale)
    rot_mat, bounds = get_rotation_mat(img, scull_center, roll)
    vertical_img = cv2.warpAffine(img, rot_mat, bounds, flags=cv2.INTER_LINEAR)
    vertices = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
    rotated_landmarks = vertices @ rot_mat.T
    return vertical_img, rotated_landmarks


def flame_params_skull_center(flame_params: FlameParams, scale: float) -> Tuple[int, int]:
    scull_center = flame_params.translation / scale
    scull_center = scull_center[0].numpy()
    return int(scull_center[0]), int(scull_center[1])


def get_rotation_mat(
    img: np.ndarray, img_center: Tuple[int, int], angle: Union[float, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = img.shape[:2]
    rotation_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - img_center[0]
    rotation_mat[1, 2] += bound_h / 2 - img_center[1]
    return rotation_mat, (bound_w, bound_h)


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


class HeadAligner:
    def __init__(self, image_pattern: str, model_name: str = "YoloHeads_M", checkpoint: str = "ckpt_best.pth"):
        self.images = glob.glob(image_pattern)
        self.model = models.get(model_name, checkpoint_path=checkpoint, num_classes=413).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _get_head_bbox(self, vertices):
        points = []
        points.extend(np.take(vertices, np.array(HEAD_INDICES), axis=0))
        points = np.array(points)
        x = min(points[:, 0])
        y = min(points[:, 1])
        x1 = max(points[:, 0])
        y1 = max(points[:, 1])
        return list(map(int, [x, y, x1 - x, y1 - y]))

    @staticmethod
    def calculate_rpy(flame_params) -> RPY:
        rot_mat = rot_mat_from_6dof(flame_params.rotation).numpy()[0]
        rot_mat_2 = np.transpose(rot_mat)
        angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
        roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
        return RPY(roll=roll, pitch=pitch, yaw=yaw)

    def predict(self, image, dsize: int = 640):
        device = infer_model_device(self.model)
        # Resize image to dsize x dsize but keep the aspect ratio by padding with zeros
        original_shape = image.shape[:2]
        # resize to dsize max side
        scale = dsize / max(original_shape)
        new_shape = (int(original_shape[1] * scale), int(original_shape[0] * scale))
        image = cv2.resize(image, new_shape)

        # Pad the image with zeros
        # For simplicity, we do bottom and right padding to simply the calculations in post-processing
        pad_w = dsize - image.shape[1]
        pad_h = dsize - image.shape[0]
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=127)

        image_input = torch.from_numpy(image).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        raw_predictions = self.model(image_input)
        (predictions,) = self.model.get_post_prediction_callback(conf=0.5, iou=0.5, post_nms_max_predictions=1)(raw_predictions)
        predictions.bboxes_xyxy /= scale
        predictions.predicted_2d_vertices /= scale  # There are 565 keypoints subset here
        predictions.predicted_3d_vertices /= scale  # There are 565 keypoints subset here
        return predictions, scale

    def align(self, image: np.ndarray, predictions, scale):
        head_images = []
        for vertices, params in zip(predictions.predicted_2d_vertices, predictions.mm_params):
            head_image = image.copy()
            flame = FlameParams.from_3dmm(params.unsqueeze(0), FLAME_CONSTS)
            pred_pose = self.calculate_rpy(flame)
            #rotate the head and vertices on roll
            roll = pred_pose.roll
            vertices = vertices.numpy()
            if np.abs(pred_pose.yaw) < 60:
                head_image, vertices = vertically_align(head_image, vertices, flame, roll, scale)
            head_bbox = self._get_head_bbox(vertices)
            head_bbox = extend_to_rect(extend_bbox(head_bbox, offset=0.1))
            #extend to rectangle
            x, y, w, h = head_bbox
            crop = head_image[y:y+h, x:x+w]
            #head_image = draw_3d_landmarks(vertices, head_image)
            head_images.append(crop)
        return head_images

    def __call__(self, *args, **kwargs):
        os.makedirs("aligned_heads", exist_ok=True)
        index = -1
        for image_path in tqdm.tqdm(self.images):
            index += 1
            image = cv2.imread(image_path)
            predictions, scale = self.predict(image)
            aligned_heads = self.align(image, predictions, scale)
            for i, head in enumerate(aligned_heads):
                cv2.imwrite(f"aligned_heads/{index}_{i}.jpg", head)


def main(
    model_name: str = "YoloHeads_L",
    checkpoint: str = "model.pth",
    image_pattern: str = './FDDB/*',
):
    aligner = HeadAligner(image_pattern=image_pattern, model_name=model_name, checkpoint=checkpoint)
    aligner()


if __name__ == "__main__":
    Fire(main)
