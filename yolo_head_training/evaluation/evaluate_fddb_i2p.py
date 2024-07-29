import os
from typing import Union
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
from fire import Fire
import cv2
from draw_utils import get_relative_path
from PIL import Image
from torchvision import transforms

import sys
from img2pose import img2poseModel
from model_loader import load_model


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
    def __init__(self, data_dir: str):
        self.dataset_dir = data_dir
        threed_68_points = np.load("img2pose/pose_references/reference_3d_68_points_trans.npy")
        self.pose_mean = np.load("WIDER/WIDER_train_pose_mean_v1.npy")
        self.pose_stddev = np.load("WIDER/WIDER_train_pose_stddev_v1.npy")
        self.model = img2poseModel(
            18,
            400,
            1400,
            pose_mean=self.pose_mean,
            pose_stddev=self.pose_stddev,
            threed_68_points=threed_68_points,
        )
        load_model(
            self.model.fpn_model,
            "img2pose_v1.pth",
            cpu_mode=str(self.model.device) == "cpu",
            model_only=True,
        )
        self.model.evaluate()
        self.annotations = self.read_annotations()
        self.transform = transforms.Compose([transforms.ToTensor()])

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

    def predict(self, image, dsize: int = 640):
        faces = self.model.predict([self.transform(Image.fromarray(image))])
        faces = faces[0]
        bboxes = []
        for i in range(len(faces["scores"])):
            bbox = list(map(int, faces["boxes"].cpu().numpy()[i]))
            score = faces["scores"].cpu().numpy()[i]
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            confidence = float(score)
            bboxes.append([x1, y1, x2 - x1, y2 - y1, confidence])
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

    def __call__(self, *args, **kwargs):
        result = {}
        index = -1
        for image_path, bboxes in tqdm.tqdm(self.annotations.items()):
            index += 1
            image = cv2.imread(os.path.join(self.dataset_dir, "images", image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictions = self.predict(image)
            result[image_path] = predictions
            gt_image = image.copy()
            #for bbox in bboxes:
            #    x, y, w, h = bbox
            #    cv2.rectangle(gt_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #for bbox in predictions:
            #    x, y, w, h, score = bbox
            #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #    cv2.putText(image, str(round(score, 2)), (x - 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #cv2.imwrite(f"fddb_test/{index}.jpg", np.hstack((gt_image, image)))
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
    fddb_path: str = "./FDDB",
):
    evaluator = FDDBEvaluator(data_dir=fddb_path)
    evaluator()


if __name__ == "__main__":
    Fire(main)
