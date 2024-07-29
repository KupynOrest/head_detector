import math
import abc
import glob
import os
from pathlib import Path
from typing import Union, Optional, Any, Tuple
import tqdm
import numpy as np
import torch
from fire import Fire
import cv2
from super_gradients.training import models
import scipy.io
from scipy.spatial.transform import Rotation
from super_gradients.training.utils.utils import infer_model_device

from yolo_head.flame import FlameParams, FLAME_CONSTS, rot_mat_from_6dof, RPY
from yolo_head.yolo_heads_predictions import YoloHeadsPredictions
from evaluation.draw_utils import draw_pose, draw_3d_landmarks, get_relative_path


FACE_INDICES = np.load(str(get_relative_path("../yolo_head/flame_indices/face.npy", __file__)),
                          allow_pickle=True)[()]


MAX_ROTATION = 99
MAE_THRESHOLD = 40


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


class HeadPoseEvaluator:
    def __init__(self, data_dir: str, model_name: str = "YoloHeads_M", checkpoint: str = "ckpt_best.pth"):
        self.dataset_dir = data_dir
        self.model = models.get(model_name, checkpoint_path=checkpoint, num_classes=413).eval()
        self.name = "pose"
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @abc.abstractmethod
    def get_gt_pose(self, label_path: str) -> Optional[Tuple[RPY, Any]]:
        pass

    @abc.abstractmethod
    def find_images_and_labels(self, dataset_dir: str):
        pass

    @abc.abstractmethod
    def select_head(self, predictions: YoloHeadsPredictions, metadata: Any) -> YoloHeadsPredictions:
        pass

    @staticmethod
    def mae(x, y):
        PI = 180.0
        return min(
            math.fabs(x - y),
            math.fabs(x - (y - 2 * PI)),
            math.fabs(x - (y + 2 * PI)),
        )

    @staticmethod
    def calculate_rpy(flame_params) -> RPY:
        rot_mat = rot_mat_from_6dof(flame_params.rotation).numpy()[0]
        rot_mat_2 = np.transpose(rot_mat)
        angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
        roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
        if np.any(np.abs([roll, pitch, yaw]) > 135):
            print("Rotation is too large")
            return RPY(roll=0, pitch=0, yaw=0)
        return RPY(roll=roll, pitch=pitch, yaw=yaw)

    def _get_face_bbox(self, vertices):
        points = []
        points.extend(np.take(vertices, np.array(FACE_INDICES), axis=0))
        points = np.array(points)
        x = min(points[:, 0])
        y = min(points[:, 1])
        x1 = max(points[:, 0])
        y1 = max(points[:, 1])
        return list(map(int, [x, y, x1, y1]))

    def predict(self, image, metadata: Any, dsize: int = 640):
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
        (predictions,) = self.model.get_post_prediction_callback(conf=0.5, iou=0.5, post_nms_max_predictions=30)(raw_predictions)
        if predictions.bboxes_xyxy.size()[0] == 0:
            (predictions,) = self.model.get_post_prediction_callback(conf=0.1, iou=0.5, post_nms_max_predictions=30)(
                raw_predictions)
        predictions.bboxes_xyxy /= scale
        predictions.predicted_2d_vertices /= scale  # There are 565 keypoints subset here
        predictions.predicted_3d_vertices /= scale  # There are 565 keypoints subset here
        if predictions.bboxes_xyxy.shape[0] > 1:
            predictions = self.select_head(predictions, metadata)

        predictions: YoloHeadsPredictions = predictions

        flame = FlameParams.from_3dmm(predictions.mm_params, FLAME_CONSTS)
        flame.scale /= scale
        predictions.mm_params = flame.to_3dmm_tensor()
        return predictions, flame

    def __call__(self, *args, **kwargs):
        os.makedirs(os.path.join("output", self.name), exist_ok=True)
        images, labels = self.find_images_and_labels(self.dataset_dir)
        metrics = {
            "roll": [],
            "pitch": [],
            "yaw": [],
        }
        fail_cases = 0
        index = -1
        for image_path, gt in tqdm.tqdm(zip(images, labels)):
            index += 1
            image = cv2.imread(str(image_path))
            ground_truth = self.get_gt_pose(str(gt))
            if ground_truth is None:
                continue
            try:
                gt_pose, metadata = ground_truth
                predictions, flame_params = self.predict(image, metadata)
                pred_pose = self.calculate_rpy(flame_params)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #gt_image = image.copy()
                #gt_image = cv2.rectangle(gt_image, tuple(metadata[:2]), tuple(metadata[2:]), (0, 255, 0), 2)
                #bbox = self._get_face_bbox(predictions.predicted_2d_vertices[0].numpy())
                #image = draw_3d_landmarks(predictions.predicted_2d_vertices.reshape(-1, 2), image)
                #image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
                #image = draw_pose(pred_pose, image)
                #gt_image = draw_pose(gt_pose, gt_image)
                #cv2.imwrite(f"output/{self.name}/{index}_pred.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                #cv2.imwrite(f"output/{self.name}/{index}_gt.jpg", cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR))
            except:
                print(f"Failed to process image {image_path}")
                fail_cases += 1
                continue
            metrics["roll"].append(self.mae(gt_pose.roll, pred_pose.roll))
            metrics["pitch"].append(self.mae(gt_pose.pitch, pred_pose.pitch))
            metrics["yaw"].append(self.mae(gt_pose.yaw, pred_pose.yaw))
        roll_mae = np.mean(np.array(metrics["roll"]))
        pitch_mae = np.mean(np.array(metrics["pitch"]))
        yaw_mae = np.mean(np.array(metrics["yaw"]))
        print(f"Roll MAE: {roll_mae}, Pitch MAE: {pitch_mae}, Yaw MAE: {yaw_mae}, MAE = {(roll_mae + pitch_mae + yaw_mae) / 3}")
        print(f"Failed cases: {fail_cases}")


class AFLWEvaluator(HeadPoseEvaluator):
    def __init__(self, data_dir: str, model_name: str = "YoloHeads_M", checkpoint: str = "ckpt_best.pth"):
        super().__init__(data_dir, model_name, checkpoint)
        self.name = "aflw"

    @staticmethod
    def calculate_iou(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        x_overlap = max(0, min(x2, x4) - max(x1, x3))
        y_overlap = max(0, min(y2, y4) - max(y1, y3))
        intersection = x_overlap * y_overlap
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        return intersection / union

    def select_head(self, predictions: YoloHeadsPredictions, metadata: Any) -> YoloHeadsPredictions:
        """
        Select the head with the highest confidence score.
        """
        head_bbox = metadata
        vertices = predictions.predicted_2d_vertices
        # select one with largest iou
        max_iou = 0
        max_index = 0
        for index, vertices_i in enumerate(vertices):
            bbox = self._get_face_bbox(vertices_i.numpy())
            iou = self.calculate_iou(bbox, head_bbox)
            if iou > max_iou:
                max_iou = iou
                max_index = index
        predictions.bboxes_xyxy = predictions.bboxes_xyxy[max_index].unsqueeze(0)
        predictions.predicted_2d_vertices = predictions.predicted_2d_vertices[max_index].unsqueeze(0)
        predictions.predicted_3d_vertices = predictions.predicted_3d_vertices[max_index].unsqueeze(0)
        predictions.scores = predictions.scores[max_index]
        predictions.mm_params = predictions.mm_params[max_index].unsqueeze(0)
        return predictions

    def find_images_and_labels(self, dataset_dir):
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

    @staticmethod
    def bbox_from_keypoints(keypoints):
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        return np.array([np.min(x), np.min(y), np.max(x), np.max(y)]).astype(int)

    def get_gt_pose(self, label_path: str) -> Optional[Tuple[RPY, Any]]:
        mat = scipy.io.loadmat(label_path)
        pose_params = mat["Pose_Para"][0]
        degrees = pose_params[:3] * (180 / np.pi)
        if np.any(np.abs(degrees) > MAX_ROTATION):
            return None
        return RPY(roll=degrees[2], pitch=degrees[0], yaw=degrees[1]), self.bbox_from_keypoints(np.asarray(mat["pt3d_68"]).T[:, :2])


class BIWIEvaluator(HeadPoseEvaluator):
    def __init__(self, data_dir: str, model_name: str = "YoloHeads_M", checkpoint: str = "ckpt_best.pth"):
        super().__init__(data_dir, model_name, checkpoint)
        self.name = "biwi"

    def find_images_and_labels(self, dataset_dir):
        images = glob.glob(f"{dataset_dir}/**/*.png")
        labels = [x.replace("rgb.png", "pose.txt") for x in images]
        return images, labels

    def get_gt_pose(self, label_path: str) -> Optional[Tuple[RPY, Any]]:
        rotation_matrix = np.loadtxt(label_path)
        rotation_matrix = rotation_matrix[:3, :]

        rotation_matrix = np.transpose(rotation_matrix)

        roll = -np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1] ** 2 + rotation_matrix[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]) * 180 / np.pi
        return RPY(roll=roll, pitch=pitch, yaw=yaw), None

    def select_head(self, predictions: YoloHeadsPredictions, metadata: Any) -> YoloHeadsPredictions:
        """
        Select the head with the highest confidence score.
        """
        bboxes = predictions.bboxes_xyxy
        # select one closes to center of image
        min_distance = 1e9
        max_index = 0
        for index, bbox in enumerate(bboxes):
            bbox_center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
            CENTER = np.array((320, 320))
            distance_to_center = np.linalg.norm(bbox_center - CENTER)
            if distance_to_center < min_distance:
                min_distance = distance_to_center
                max_index = index
        predictions.bboxes_xyxy = predictions.bboxes_xyxy[max_index].unsqueeze(0)
        predictions.predicted_2d_vertices = predictions.predicted_2d_vertices[max_index].unsqueeze(0)
        predictions.predicted_3d_vertices = predictions.predicted_3d_vertices[max_index].unsqueeze(0)
        predictions.scores = predictions.scores[max_index]
        predictions.mm_params = predictions.mm_params[max_index].unsqueeze(0)
        return predictions


def main(
    model_name: str = "YoloHeads_L",
    checkpoint: str = "C:\Develop\GitHub\VGG\head_detector\yolo_head_training\weights\yolo_heads_l_ckpt_best_nme_1.562.pth",
    aflw_dir: str = "g:/AFLW2000",
    biwi_dir: Optional[str] = None,
):
    evaluators = [AFLWEvaluator(data_dir=aflw_dir, model_name=model_name, checkpoint=checkpoint)]
    if biwi_dir is not None:
        evaluators.append(BIWIEvaluator(data_dir=biwi_dir, model_name=model_name, checkpoint=checkpoint))
    for evaluator in evaluators:
        evaluator()


if __name__ == "__main__":
    Fire(main)
