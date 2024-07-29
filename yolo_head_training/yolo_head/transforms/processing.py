from typing import Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
from super_gradients.common.registry.registry import register_processing
from super_gradients.training.processing.processing import _LongestMaxSizeRescale, RescaleMetadata, Processing, DetectionPadToSizeMetadata, PaddingCoordinates
from super_gradients.training.transforms.utils import _rescale_bboxes, _shift_bboxes_xyxy, _pad_image, _shift_keypoints, _get_bottom_right_padding_coordinates

from ..yolo_heads_predictions import YoloHeadsPredictions
from ..flame import FlameParams, FLAME_CONSTS


@register_processing("MeshLongestMaxSizeRescale")
class MeshLongestMaxSizeRescale(_LongestMaxSizeRescale):
    def postprocess_predictions(self, predictions: YoloHeadsPredictions, metadata: RescaleMetadata) -> YoloHeadsPredictions:
        predictions.predicted_2d_vertices /= metadata.scale_factor_h
        predictions.predicted_3d_vertices /= metadata.scale_factor_h
        flame = FlameParams.from_3dmm(predictions.mm_params, FLAME_CONSTS)
        flame.scale /= metadata.scale_factor_h
        predictions.mm_params = flame.to_3dmm_tensor()
        if predictions.bboxes_xyxy is not None:
            predictions.bboxes_xyxy = _rescale_bboxes(targets=predictions.bboxes_xyxy, scale_factors=(1 / metadata.scale_factor_h, 1 / metadata.scale_factor_w))
        return predictions


class _MeshPadding(Processing, ABC):
    """Base class for keypoints padding methods. One should implement the `_get_padding_params` method to work with a custom padding method.

    Note: This transformation assume that dimensions of input image is equal or less than `output_shape`.

    :param output_shape: Output image shape (H, W)
    :param pad_value:   Padding value for image
    """

    def __init__(self, output_shape: Tuple[int, int], pad_value: int):
        self.output_shape = tuple(output_shape)
        self.pad_value = pad_value

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, DetectionPadToSizeMetadata]:
        padding_coordinates = self._get_padding_params(input_shape=image.shape)
        processed_image = _pad_image(image=image, padding_coordinates=padding_coordinates, pad_value=self.pad_value)
        return processed_image, DetectionPadToSizeMetadata(padding_coordinates=padding_coordinates)

    def postprocess_predictions(self, predictions: YoloHeadsPredictions, metadata: DetectionPadToSizeMetadata) -> YoloHeadsPredictions:
        predictions.predicted_2d_vertices = _shift_keypoints(
            targets=predictions.predicted_2d_vertices,
            shift_h=-metadata.padding_coordinates.top,
            shift_w=-metadata.padding_coordinates.left,
        )
        predictions.predicted_3d_vertices = _shift_keypoints(
            targets=predictions.predicted_3d_vertices,
            shift_h=-metadata.padding_coordinates.top,
            shift_w=-metadata.padding_coordinates.left,
        )
        flame = FlameParams.from_3dmm(predictions.mm_params, FLAME_CONSTS)
        flame.translation[..., 0] = flame.translation[..., 0] - metadata.padding_coordinates.left
        flame.translation[..., 1] = flame.translation[..., 1] - metadata.padding_coordinates.top
        predictions.mm_params = flame.to_3dmm_tensor()
        if predictions.bboxes_xyxy is not None:
            predictions.bboxes_xyxy = _shift_bboxes_xyxy(
                targets=predictions.bboxes_xyxy,
                shift_h=-metadata.padding_coordinates.top,
                shift_w=-metadata.padding_coordinates.left,
            )
        return predictions

    @abstractmethod
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        pass

    def get_equivalent_photometric_module(self) -> Optional[nn.Module]:
        return None

    def infer_image_input_shape(self) -> Optional[Tuple[int, int]]:
        """
        Infer the output image shape from the processing.

        :return: (rows, cols) Returns the last known output shape for all the processings.
        """
        return self.output_shape

    @property
    def resizes_image(self) -> bool:
        return True


@register_processing("MeshBottomRightPadding")
class MeshBottomRightPadding(_MeshPadding):
    def _get_padding_params(self, input_shape: Tuple[int, int]) -> PaddingCoordinates:
        return _get_bottom_right_padding_coordinates(input_shape=input_shape, output_shape=self.output_shape)
