from dataclasses import dataclass
from typing import List

import cv2
import insightface
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

IMAGE_SIZE = 640


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def iou(self, other: "Box") -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        iou = intersection_area / (
                    (self.x2 - self.x1) * (self.y2 - self.y1) + (other.x2 - other.x1) * (
                        other.y2 - other.y1) - intersection_area)
        return iou

    def to_xywh(self):
        return np.array([self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1])


class HeadDetector:
    def __init__(self, model_path: str, threshold: float = 0.5):
        providers = ["CUDAExecutionProvider"]
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 8
        sess_opt.add_session_config_entry('session.intra_op_thread_affinities',
                                          '3,4;5,6;7,8;9,10;11,12;13,14;15,16')  # set affinities of all 7 threads to cores in the first NUMA node
        self.model = ort.InferenceSession(model_path, sess_opt, providers=providers)
        self.threshold = threshold
        self.size = np.array([[IMAGE_SIZE, IMAGE_SIZE]])

    @staticmethod
    def _rescale_bbox(bbox: np.ndarray, original_shape: np.ndarray, resized_shape: np.ndarray) -> np.ndarray:
        bbox = bbox.copy()
        bbox[0] = bbox[0] * original_shape[1] / resized_shape[0]
        bbox[1] = bbox[1] * original_shape[0] / resized_shape[1]
        bbox[2] = bbox[2] * original_shape[1] / resized_shape[0]
        bbox[3] = bbox[3] * original_shape[0] / resized_shape[1]
        return np.array(bbox).astype("int")

    def _nms(self, bboxes: List[Box], iou_threshold: float = 0.5) -> List[Box]:
        bboxes = sorted(bboxes, key=lambda x: x.confidence, reverse=True)
        result = []
        for box in bboxes:
            if all(box.iou(result_box) < iou_threshold for result_box in result):
                result.append(box)
        return result

    def __call__(self, image: np.ndarray) -> List[Box]:
        result = []
        original_shape = image.shape
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LANCZOS4)
        image = Image.fromarray(image)
        image = ToTensor()(image)[None]
        output = self.model.run(
            output_names=None,
            input_feed={'images': image.data.numpy(),
                        "orig_target_sizes": np.ascontiguousarray(self.size).astype(np.int64)}
        )
        all_labels, all_boxes, all_scores = output
        for i in range(image.shape[0]):
            scr = all_scores[i]
            boxes = all_boxes[i][scr > self.threshold]
            for index, box in enumerate(boxes):
                box = self._rescale_bbox(box, np.array(original_shape), self.size[0])
                result.append(Box(*box, scr[index]))
        return self._nms(result)


class FaceDetector:
    def __init__(self):
        self.detector = insightface.app.FaceAnalysis()
        self.detector.prepare(ctx_id=0)

    def __call__(self, image: np.ndarray) -> List[Box]:
        faces = self.detector.get(image)
        result = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            confidence = face.norm_score
            result.append(Box(x1, y1, x2, y2, confidence))
        return result
