import os
import json
from typing import List, Tuple, Optional

import cv2
from fire import Fire
from glob import glob
from tqdm import tqdm
import numpy as np

from binary_detector import HeadDetector, FaceDetector, Box


#ToDo: Remove from final version
def viz_bbox(image: np.ndarray, bboxes: List[Box], color: Tuple[int, int, int]) -> np.ndarray:
    image = np.array(image).astype(np.uint8)
    for bbox in bboxes:
        cv2.rectangle(image, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), color, 2)
    return image


def fliplr_boxes(bboxes: List[Box], image_shape: Tuple[int, int]) -> List[Box]:
    return [Box(image_shape[1] - bbox.x2, bbox.y1, image_shape[1] - bbox.x1, bbox.y2, confidence=bbox.confidence) for bbox in bboxes]


class StabilityMetric:
    def __init__(self, detector: HeadDetector):
        self.detector = detector

    def _compute_iou(self, bboxes: List[Box], flipped_bboxes: List[Box]) -> float:
        iou = 0
        for bbox in bboxes:
            iter_iou = 0
            for flipped_bbox in flipped_bboxes:
                cur_iou = bbox.iou(flipped_bbox)
                iter_iou = max(iter_iou, cur_iou)
            iou += iter_iou
        return iou / len(bboxes)

    def __call__(self, image: np.ndarray) -> Tuple[int, float]:
        bboxes = self.detector(image)
        filpped_bboxes = self.detector(image)
        filpped_bboxes = fliplr_boxes(filpped_bboxes, image.shape)
        return len(bboxes), self._compute_iou(bboxes, filpped_bboxes)


class DetectorFilter:
    def __init__(self, detector: HeadDetector):
        self.detector = detector

    def __call__(self, image: np.ndarray) -> bool:
        bboxes = self.detector(image)
        if len(bboxes) == 0:
            return True
        image = np.fliplr(image)
        filpped_bboxes = self.detector(image)
        filpped_bboxes = fliplr_boxes(filpped_bboxes, image.shape)
        if len(bboxes) != len(filpped_bboxes):
            return True
        return False

    #ToDo: Remove from final version
    def visualize(self, image: np.ndarray) -> np.ndarray:
        bboxes = self.detector(image)
        if len(bboxes) == 0:
            return image
        image = np.fliplr(image)
        filpped_bboxes = self.detector(image)
        image = np.fliplr(image)
        filpped_bboxes = fliplr_boxes(filpped_bboxes, image.shape)
        image = viz_bbox(image, bboxes, (255, 0, 0))
        image = viz_bbox(image, filpped_bboxes, (0, 0, 255))
        return image


class VerticalCutFilter:
    def __init__(self, detector: HeadDetector):
        self.detector = detector

    @staticmethod
    def find_vertical_split(image_width, bboxes: List[Box]) -> Optional[int]:
        center_x = image_width // 2
        for offset in range(center_x + 1):
            left_x = center_x - offset
            right_x = center_x + offset

            # Check if the left split line intersects with any bounding boxes
            if any(box.x1 < left_x < box.x2 for box in bboxes):
                # If left split line intersects, return the right split line if it doesn't intersect
                if not any(box.x1 < right_x < box.x2 for box in bboxes):
                    return right_x
            # If the left split line doesn't intersect, return it
            else:
                return left_x
        # If no non-intersecting split line is found, return None
        return None

    def __call__(self, image: np.ndarray) -> bool:
        bboxes = self.detector(image)
        if len(bboxes) == 0:
            return True
        vertical_split = self.find_vertical_split(image.shape[1], bboxes)
        image_l = image[:, :vertical_split]
        image_r = image[:, vertical_split:]
        if image_l.shape[1] < 10 or image_r.shape[1] < 10 or image_l.shape[0] < 10 or image_r.shape[0] < 10:
            return True
        bboxes_l = self.detector(image_l)
        bboxes_r = self.detector(image_r)
        if len(bboxes_l) + len(bboxes_r) != len(bboxes):
            return True
        return False

    # ToDo: Remove from final version
    def visualize(self, image: np.ndarray) -> np.ndarray:
        bboxes = self.detector(image)
        if len(bboxes) == 0:
            return image
        vertical_split = self.find_vertical_split(image.shape[1], bboxes)
        image_l = image[:, :vertical_split]
        image_r = image[:, vertical_split:]
        bboxes_l = self.detector(image_l)
        bboxes_r = self.detector(image_r)
        bboxes_r = [Box(bbox.x1 + vertical_split, bbox.y1, bbox.x2 + vertical_split, bbox.y2, confidence=bbox.confidence) for bbox in bboxes_r]
        image = viz_bbox(image, bboxes, (255, 0, 0))
        image = viz_bbox(image, bboxes_l, (0, 0, 255))
        image = viz_bbox(image, bboxes_r, (0, 255, 0))
        #Draw vertical line
        cv2.line(image, (vertical_split, 0), (vertical_split, image.shape[0]), (0, 255, 255), 2)
        return image


class FaceDetectorFilter:
    def __init__(self, detector: HeadDetector):
        self.detector = detector
        self.face_detector = FaceDetector()

    def _bbox_overlap(self, bbox1: Box, bbox2: Box) -> bool:
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        return intersection_area > 0

    def _match_face(self, face_bbox: Box, head_bboxes: List[Box]) -> bool:
        for head_bbox in head_bboxes:
            if self._bbox_overlap(face_bbox, head_bbox):
                return True
        return False

    def __call__(self, image: np.ndarray) -> bool:
        heads = self.detector(image.copy())
        faces = self.face_detector(image)
        if len(faces) == 0:
            return False
        for face_bbox in faces:
            if not self._match_face(face_bbox, heads):
                return True
        return False

    #ToDo: Remove from final version
    def visualize(self, image: np.ndarray) -> np.ndarray:
        faces = self.face_detector(image)
        heads = self.detector(image)
        if len(faces) == 0:
            return image
        image = viz_bbox(image, faces, (0, 255, 0))
        image = viz_bbox(image, heads, (0, 0, 255))
        return image


def filter_data(data_path: str, head_detector_path: str, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    detector = HeadDetector(head_detector_path)
    single_image_filters = [DetectorFilter(detector), FaceDetectorFilter(detector), VerticalCutFilter(detector)]
    stability_metric = StabilityMetric(detector)
    images = glob(f"{data_path}/*.jpg")
    metrics = []
    for image_path in tqdm(images):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        skip = False
        for filter in single_image_filters:
            if filter(image):
                viz = filter.visualize(image)
                cv2.imwrite(os.path.join(save_path, f"viz_{os.path.basename(image_path)}"), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                skip = True
                break
        if skip:
            continue
        num_heads, metric = stability_metric(image)
        metrics.append({"filename": image_path, "num_heads": num_heads, "stability_metric": metric})
    with open(os.path.join(save_path, "metrics.json"), "w") as file:
        json.dump(metrics, file)


if __name__ == "__main__":
    Fire(filter_data)
