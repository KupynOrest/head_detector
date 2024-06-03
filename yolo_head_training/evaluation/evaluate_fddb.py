import os
from typing import Union
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import torch
from fire import Fire
import cv2
from super_gradients.training import models
from super_gradients.training.utils.utils import infer_model_device

from scipy.spatial.transform import Rotation
from yolo_head.flame import FlameParams, FLAME_CONSTS, rot_mat_from_6dof, RPY
from draw_utils import get_relative_path


FACE_INDICES = np.load(str(get_relative_path("../yolo_head/flame_indices/face.npy", __file__)),
                          allow_pickle=True)[()]


MAX_ROTATION = 110
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


class FDDBEvaluator:
    def __init__(self, data_dir: str, model_name: str = "YoloHeads_M", checkpoint: str = "ckpt_best.pth"):
        self.dataset_dir = data_dir
        self.model = models.get(model_name, checkpoint_path=checkpoint, num_classes=413).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.annotations = self.read_annotations()

    def read_annotations(self):
        with open(os.path.join(self.dataset_dir, "label.txt"), 'r') as f:
            lines = f.readlines()
        annotations = {}
        current_image = None
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                current_image = line[2:]  # Remove the '# ' part
                annotations[current_image] = []
            else:
                x, y, x1, y1 = list(map(int, line.split()))
                annotations[current_image].append([x, y, x1 - x, y1 - y])

        return annotations

    def _get_face_bbox(self, vertices):
        points = []
        points.extend(np.take(vertices, np.array(FACE_INDICES), axis=0))
        points = np.array(points)
        x = min(points[:, 0])
        y = min(points[:, 1])
        x1 = max(points[:, 0])
        y1 = max(points[:, 1])
        return list(map(int, [x, y, x1 - x, y1 - y]))

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
        (predictions,) = self.model.get_post_prediction_callback(conf=0.5, iou=0.5, post_nms_max_predictions=100)(raw_predictions)
        predictions.bboxes_xyxy /= scale
        predictions.predicted_2d_vertices /= scale  # There are 565 keypoints subset here
        predictions.predicted_3d_vertices /= scale  # There are 565 keypoints subset here
        predictions = self.parse_predictions(predictions, image)
        return predictions

    @staticmethod
    def calculate_rpy(flame_params) -> RPY:
        rot_mat = rot_mat_from_6dof(flame_params.rotation).numpy()[0]
        rot_mat_2 = np.transpose(rot_mat)
        angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
        roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
        return RPY(roll=roll, pitch=pitch, yaw=yaw)

    @staticmethod
    def _valid_vertices(vertices, image_shape):
        return np.sum((vertices[:, 0] >= 0) & (vertices[:, 0] < image_shape[1]) & (vertices[:, 1] >= 0) & (vertices[:, 1] < image_shape[0])) > 0.5 * len(vertices)

    def parse_predictions(self, predictions, image):
        bboxes = []
        for index, (vertices_i, params) in enumerate(zip(predictions.predicted_2d_vertices, predictions.mm_params)):
            flame = FlameParams.from_3dmm(params.unsqueeze(0), FLAME_CONSTS)
            if not self._valid_vertices(vertices_i.numpy(), image.shape[:2]):
                continue
            pred_pose = self.calculate_rpy(flame)
            if abs(pred_pose.yaw) > MAX_ROTATION or (abs(pred_pose.roll) > MAX_ROTATION and abs(pred_pose.pitch) > MAX_ROTATION):
                continue
            bbox = self._get_face_bbox(vertices_i.numpy())
            score = float(predictions.scores[index])
            bboxes.append([*bbox, score])
        return bboxes

    def convert_to_coco_format(self, gt_dict, pred_dict):
        gt_coco_format = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}]
        }
        pred_coco_format = []

        annotation_id = 1
        image_id_map = {}
        current_image_id = 1

        # Convert ground truth
        for filename, bboxes in gt_dict.items():
            image_id = current_image_id
            image_id_map[filename] = image_id
            gt_coco_format["images"].append({"id": image_id, "file_name": filename})

            for bbox in bboxes:
                x_min, y_min, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
                gt_coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x_min, y_min, width, height],
                    "area": width * height,
                    "iscrowd": 0
                })
                annotation_id += 1

            current_image_id += 1

        # Convert predictions
        annotation_id = 1
        for filename, bboxes in pred_dict.items():
            image_id = image_id_map[filename]

            for bbox in bboxes:
                x_min, y_min, width, height, score = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
                pred_coco_format.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x_min, y_min, width, height],
                    "score": score
                })
                annotation_id += 1

        return gt_coco_format, pred_coco_format

    def sanitize_predictions(self, bboxes, image):
        cropped_boxes = []
        for bbox in bboxes:
            x1, y1, w, h, score = bbox
            x1_new, y1_new = min(max(0, x1), image.shape[1]), min(max(0, y1), image.shape[0])
            w = w - (x1_new - x1)
            h = h - (y1_new - y1)
            x2, y2 = min(max(0, x1_new + w), image.shape[1]), min(max(0, y1_new + h), image.shape[0])
            w, h = x2 - x1_new, y2 - y1_new
            cropped_boxes.append([x1_new, y1_new, w, h, score])
        return cropped_boxes

    def __call__(self, *args, **kwargs):
        result = {}
        index = -1
        for image_path, bboxes in tqdm.tqdm(self.annotations.items()):
            index += 1
            image = cv2.imread(os.path.join(self.dataset_dir, "images", image_path))
            predictions = self.predict(image)
            predictions = self.sanitize_predictions(predictions, image)
            result[image_path] = predictions
            gt_image = image.copy()
            for bbox in bboxes:
                x, y, w, h = bbox
                cv2.rectangle(gt_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            for bbox in predictions:
                x, y, w, h, score = bbox
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, str(round(score, 2)), (x - 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(f"fddb_test/{index}.jpg", np.hstack((gt_image, image)))
        gt_coco_format, pred_coco_format = self.convert_to_coco_format(self.annotations, result)

        # Save the ground truth and predictions in json files
        with open('gt_coco_format.json', 'w') as f:
            json.dump(gt_coco_format, f)

        with open('pred_coco_format.json', 'w') as f:
            json.dump(pred_coco_format, f)

        # Load COCO ground truth and predictions
        coco_gt = COCO('gt_coco_format.json')
        coco_dt = coco_gt.loadRes('pred_coco_format.json')

        # Evaluate
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Print mAP
        print("mAP: {:.2f}".format(coco_eval.stats[0] * 100))


def main(
    model_name: str = "YoloHeads_L",
    checkpoint: str = "C:\Develop\GitHub\VGG\head_detector\yolo_head_training\weights\yolo_heads_l_ckpt_best_nme_1.562.pth",
    fddb_path: str = "/work/okupyn/FDDB",
):
    evaluator = FDDBEvaluator(data_dir=fddb_path, model_name=model_name, checkpoint=checkpoint)
    evaluator()


if __name__ == "__main__":
    Fire(main)
