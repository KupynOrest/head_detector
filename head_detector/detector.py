from typing import List, Union, Tuple, Dict, Any

import cv2
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from head_detector.head_info import HeadMetadata, Bbox
from head_detector.detection_result import PredictionResult
from head_detector.utils import nms, calculate_rpy
from head_detector.flame import FlameParams, FLAMELayer, reproject_spatial_vertices


REPO_ID = "okupyn/vgg_heads"


class HeadDetector:
    def __init__(self, model: str = "vgg_heads_l", image_size: int = 640):
        self._image_size = image_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._flame = FLAMELayer().to(self._device)
        self.model = self._read_model(model)

    def _read_model(self, model: str) -> torch.jit.ScriptModule:
        #ToDo: Proper serialization for cpu
        if not torch.cuda.is_available():
            model = f"{model}_cpu"
        model_path = hf_hub_download(REPO_ID, f"{model}.trcd")
        loaded_model = torch.jit.load(model_path)
        loaded_model.to(self._device)
        loaded_model.eval()
        return loaded_model

    def _convert_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        return image

    def _transform_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], float]:
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self._image_size, int(w * self._image_size / h)
        else:
            new_h, new_w = int(h * self._image_size / w), self._image_size
        scale = self._image_size / max(image.shape[:2])
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        pad_w = self._image_size - image.shape[1]
        pad_h = self._image_size - image.shape[0]
        image = cv2.copyMakeBorder(image, pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2, cv2.BORDER_CONSTANT, value=127)
        image_input = torch.from_numpy(image).to(self._device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return image_input, (pad_w // 2, pad_h // 2), scale

    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image, padding, scale = self._transform_image(image)
        return image, {"padding": padding, "scale": scale}

    def _process(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(image)

    def _parse_predictions(self, bboxes_xyxy: torch.Tensor, scores: torch.Tensor, flame_params: torch.Tensor, cache: Dict[str, Any]):
        padding = cache["padding"]
        scale = cache["scale"]
        bboxes_xyxy = bboxes_xyxy.cpu().numpy()
        scores = scores.cpu().numpy()
        _, _, final_3d_pts = reproject_spatial_vertices(self._flame, flame_params, to_2d=False)
        final_3d_pts[:, :, 0] -= padding[0]
        final_3d_pts[:, :, 1] -= padding[1]
        final_3d_pts = (final_3d_pts / scale).cpu().numpy()
        bboxes_xyxy = bboxes_xyxy.clip(0, self._image_size)
        bboxes_xyxy[:, [0, 2]] -= padding[0]
        bboxes_xyxy[:, [1, 3]] -= padding[1]
        bboxes_xyxy /= scale
        bboxes_xyxy = np.rint(bboxes_xyxy).astype(int)
        result = []
        flame_params = flame_params.detach().cpu()
        for bbox, score, params, vertices in zip(bboxes_xyxy, scores, flame_params, final_3d_pts):
            params = FlameParams.from_3dmm(params.unsqueeze(0))
            box = Bbox(x=bbox[0], y=bbox[1], w=bbox[2] - bbox[0], h=bbox[3] - bbox[1])
            result.append(
                HeadMetadata(
                    bbox=box,
                    score=score,
                    flame_params=params,
                    vertices_3d=vertices,
                    head_pose=calculate_rpy(params)
                )
            )
        return result

    def _postprocess(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], cache: Dict[str, Any], confidence_threshold: float) -> List[HeadMetadata]:
        boxes, scores, flame_params = predictions
        boxes, scores, flame_params = nms(boxes, scores, flame_params, confidence_threshold=confidence_threshold)
        return self._parse_predictions(boxes, scores, flame_params, cache)

    def __call__(self, image: Union[str, Image.Image, np.ndarray], confidence_threshold: float = 0.5) -> PredictionResult:
        original_image = self._convert_image(image)
        image, cache = self._preprocess(original_image)
        predictions = self._process(image)
        heads = self._postprocess(predictions, cache, confidence_threshold)
        return PredictionResult(original_image=original_image, heads=heads)
