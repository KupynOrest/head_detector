import os
import cv2
import tqdm
import json
import math
import torch
from typing import Dict, Any, List, Tuple
from collections import namedtuple, defaultdict
import numpy as np
from head_mesh import HeadMesh
from yolo_head.flame import FlameParams, FLAME_CONSTS, rot_mat_from_6dof
from super_gradients.training.utils.utils import infer_model_device

from scipy.spatial.transform import Rotation

from evaluation.dad_utils import get_68_landmarks, calc_zn, calc_ch_dist
from super_gradients.training import models
from fire import Fire
from evaluation.draw_utils import draw_3d_landmarks, get_relative_path


HEAD_INDICES = np.load(str(get_relative_path("../yolo_head/flame_indices/head_indices.npy", __file__)),
                          allow_pickle=True)[()]

MeshArrays = namedtuple(
    "MeshArrays",
    ["vertices3d", "vertices3d_world_homo", "projection_matrix", "model_view_matrix"],
)


class HeadAnnotation:
    """Class for storing a head annotation."""

    def __init__(self, image_path: str, mesh: MeshArrays, bbox: List[int], attributes: Dict[str, str]):
        self.image_path = image_path
        self.mesh = mesh
        self.bbox = bbox
        self.attributes = attributes

    def landmarks_2d(self, image_shape: Tuple[int, int]) -> np.ndarray:
        flame_vertices2d_homo = np.transpose(
            np.matmul(self.mesh.projection_matrix, np.transpose(self.mesh.vertices3d_world_homo))
        )
        flame_vertices2d = flame_vertices2d_homo[:, :2] / flame_vertices2d_homo[:, [3]]
        keypoints_2d = np.stack((flame_vertices2d[:, 0], (image_shape[0] - flame_vertices2d[:, 1])), -1)
        return keypoints_2d

    def landmarks_68_2d(self, image_shape: Tuple[int, int]) -> np.ndarray:
        landmarks = get_68_landmarks(torch.from_numpy(self.mesh.vertices3d).view(-1, 3)).numpy()
        landmarks = np.concatenate((landmarks, np.ones_like(landmarks[:, [0]])), -1)
        # rotated and translated (to world coordinates)
        landmarks = np.transpose(np.matmul(self.mesh.model_view_matrix, np.transpose(landmarks)))
        landmarks = np.transpose(np.matmul(self.mesh.projection_matrix, np.transpose(landmarks)))
        landmarks = landmarks[:, :2] / landmarks[:, [3]]
        landmarks = np.stack((landmarks[:, 0], (image_shape[0] - landmarks[:, 1])), -1)
        return landmarks

    @classmethod
    def from_config(cls, config: Dict[str, Any], base_path: str = "/mnt/pinatanas/pf-cv-datasets") -> "HeadAnnotation":
        with open(os.path.join(base_path, config["json_path"])) as json_data:
            mesh_data = json.load(json_data)
        flame_vertices3d = np.array(mesh_data["vertices"], dtype=np.float32)
        model_view_matrix = np.array(mesh_data["model_view_matrix"], dtype=np.float32)
        flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
        # rotated and translated (to world coordinates)
        flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))
        mesh = MeshArrays(
            vertices3d=flame_vertices3d,
            vertices3d_world_homo=flame_vertices3d_world_homo,  # with pose and translation
            projection_matrix=np.array(mesh_data["projection_matrix"], dtype=np.float32),
            model_view_matrix=np.array(mesh_data["model_view_matrix"], dtype=np.float32),
        )
        #xywh to xyxy
        x, y, w, h = config["bbox"]
        bbox = [x, y, x + w, y + h]

        return cls(
            image_path=config["img_path"],
            mesh=mesh,
            bbox=bbox,
            attributes=config["attributes"],
        )


class PinataEvaluator:
    def __init__(
        self,
        dataset_path: str,
        predictor,
        base_path: str = "/mnt/pinatanas/pf-cv-datasets",
        img_size: int = 256,
    ):
        self._dataset_path = dataset_path
        self.predictor = predictor
        self.model_name = self.predictor.model_name if hasattr(self.predictor, "model_name") else "model"
        self.base_path = base_path
        self.test_samples = self._get_data()
        self.head_mesh = HeadMesh(flame_config=FLAME_CONSTS, image_size=img_size)
        self._img_size = img_size
        self.head_indices = np.load("/home/okupyn/head_detector/yolo_head_training/yolo_head/flame_indices/head.npy")
        self.attribute_metrics = {
            "quality": defaultdict(dict),
            "gender": defaultdict(dict),
            "expression": defaultdict(dict),
            "age": defaultdict(dict),
            "occlusions": defaultdict(dict),
            "pose": defaultdict(dict),
            "standard light": defaultdict(dict),
        }
        self.metrics = {"nme_2d": [], "z_n": [], "rot_error": [], "angle_error": [], "chamfer": []}

    @staticmethod
    def mae(x, y):
        PI_2 = 90.0
        return min(
            math.fabs(x - y),
            math.fabs(x - (y - 2 * PI_2)),
            math.fabs(x - (y + 2 * PI_2)),
        )

    @staticmethod
    def read_img(path: str) -> np.ndarray:
        img = cv2.imread(path)
        return img

    def _get_data(self) -> List[HeadAnnotation]:
        annotations = []
        with open(self._dataset_path) as json_file:
            data = json.load(json_file)
        for dataset_name, data_values in data.items():
            annotations += data_values
        return [HeadAnnotation.from_config(config=anno, base_path=self.base_path) for anno in annotations]

    def _register_attribute_metric(
        self, attribute_name: str, attribute_value: str, metric_name: str, metric_value: float
    ) -> None:
        if metric_name not in self.attribute_metrics[attribute_name][attribute_value]:
            self.attribute_metrics[attribute_name][attribute_value][metric_name] = [metric_value]
        else:
            self.attribute_metrics[attribute_name][attribute_value][metric_name].append(metric_value)

    def add_metric(self, metric_name: str, metric_value: float, attributes: Dict[str, Any]) -> None:
        self.metrics[metric_name].append(metric_value)
        for attribute_key, attribute_value in attributes.items():
            self._register_attribute_metric(
                attribute_name=attribute_key,
                attribute_value=str(attribute_value),
                metric_name=metric_name,
                metric_value=metric_value,
            )

    def _get_face_bbox(self, vertices):
        points = []
        points.extend(np.take(vertices, np.array(HEAD_INDICES), axis=0))
        points = np.array(points)
        x = min(points[:, 0])
        y = min(points[:, 1])
        x1 = max(points[:, 0])
        y1 = max(points[:, 1])
        return list(map(int, [x, y, x1, y1]))

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

    def select_head(self, predictions, metadata):
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

    def get_predictions(self, image, bbox, dsize: int = 640):
        device = infer_model_device(self.predictor)
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

        result = {}
        raw_predictions = self.predictor(image_input)
        (predictions,) = self.predictor.get_post_prediction_callback(conf=0.2, iou=0.5, post_nms_max_predictions=30)(
            raw_predictions)
        if predictions.bboxes_xyxy.size()[0] == 0:
            return None, None
        predictions.bboxes_xyxy /= scale
        predictions.predicted_2d_vertices /= scale  # There are 565 keypoints subset here
        predictions.predicted_3d_vertices /= scale  # There are 565 keypoints subset here
        if predictions.bboxes_xyxy.shape[0] > 1:
            predictions = self.select_head(predictions, bbox)
        else:
            predictions.bboxes_xyxy = predictions.bboxes_xyxy[0].unsqueeze(0)
            predictions.predicted_2d_vertices = predictions.predicted_2d_vertices[0].unsqueeze(0)
            predictions.predicted_3d_vertices = predictions.predicted_3d_vertices[0].unsqueeze(0)
            predictions.scores = predictions.scores[0]
            predictions.mm_params = predictions.mm_params[0].unsqueeze(0)
        flame = FlameParams.from_3dmm(predictions.mm_params, FLAME_CONSTS)
        flame.scale /= scale
        predictions.mm_params = flame.to_3dmm_tensor()
        return predictions, flame

    def __call__(self, *args, **kwargs) -> Dict[str, float]:
        index = -1
        fail_cases = 0
        for annotation in tqdm.tqdm(self.test_samples):
            attributes = annotation.attributes
            image = self.read_img(os.path.join(self.base_path, annotation.image_path))
            index += 1
            predictions, flame_params = self.get_predictions(image, annotation.bbox)
            if predictions is None:
                fail_cases += 1
                continue
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = draw_3d_landmarks(predictions.predicted_2d_vertices.reshape(-1, 2), image)
            #cv2.imwrite(f"test_dad/{index}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            gt_vertices_68_2d = annotation.landmarks_68_2d(image_shape=image.shape[:2])
            rotation_mat = rot_mat_from_6dof(flame_params.rotation)[0].numpy()
            rot_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            mv = rot_180 @ annotation.mesh.model_view_matrix
            R_KT = mv[:3, :3]
            R_dist = rotation_mat @ R_KT.T
            rot_error = np.linalg.norm(np.eye(3) - R_dist, "fro")
            self.add_metric("rot_error", rot_error, attributes)

            # yields min(|err|, |err-180|, |err+180}), because axis-angle representation is accurate up to 180 degrees.
            angle = self.mae(0.0, np.rad2deg(np.linalg.norm(Rotation.from_matrix(R_dist).as_rotvec())))
            self.add_metric("angle_error", angle, attributes)

            # endregion

            # region NME Calculation
            projected_vertices = predictions.predicted_3d_vertices

            landmarks_2d = get_68_landmarks(projected_vertices[0])[..., :2]
            landmarks_2d = landmarks_2d.detach().cpu().numpy()

            #landmarks_2d += np.array([[crop_point[0], crop_point[1]]])
            nme_2d = (
                float(
                    np.mean(
                        np.linalg.norm(gt_vertices_68_2d - landmarks_2d, 2, -1)
                        / (np.sqrt(annotation.bbox[2] * annotation.bbox[3]))
                    )
                )
                * 100.0
            )
            self.add_metric("nme_2d", nme_2d, attributes)

            # endregion

            # region z_n calculation
            pred_vertices = predictions.predicted_3d_vertices
            pred_vertices_head = pred_vertices[:, self.head_indices]
            gt_vertices_3d = torch.from_numpy(annotation.mesh.vertices3d_world_homo[:, :3]).view(-1, 3)
            gt_vertices_3d_head = gt_vertices_3d[self.head_indices] * -1
            z_n = calc_zn(
                pred_vertices_head.view(1, -1, 3),
                gt_vertices_3d_head.view(1, -1, 3),
            )
            self.add_metric("z_n", z_n, attributes)
            # endregion

            # region chamfer distance
            ch = calc_ch_dist(
                 gt_vertices_3d, pred_vertices.view(-1, 3), pred_lmks=get_68_landmarks(pred_vertices.view(-1, 3))
            )
            self.add_metric("chamfer", ch, attributes)

            # endregion
        print(f"Num Fail Cases = {fail_cases}")
        return {
            "nme_2d": float(np.mean(np.array(self.metrics["nme_2d"]))),
            "z_n": np.mean(np.array(self.metrics["z_n"])),
            "rot_error": np.mean(np.array(self.metrics["rot_error"])),
            "angle_error": np.mean(np.array(self.metrics["angle_error"])),
            "chamfer": np.mean(np.array(self.metrics["chamfer"])),
        }


def main(
    model_name: str = "YoloHeads_L",
    checkpoint: str = "",
):
    model = models.get(model_name, checkpoint_path=checkpoint, num_classes=413).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    evaluator = PinataEvaluator(
        dataset_path="/mnt/pinatanas/pf-cv-datasets/head_landmarks/head_landmarks/camera_ready_data_150322/anno_acad_test_camera_ready.json",
        predictor=model,
    )
    metrics = evaluator()
    print(metrics)


if __name__ == "__main__":
    Fire(main)
