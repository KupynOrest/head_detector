import dataclasses
import numpy as np
import torch

from typing import Optional, List, Union

from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh

__all__ = ["MeshEstimationSample"]

from super_gradients.training.utils.detection_utils import change_bbox_bounds_for_image_size_inplace


@dataclasses.dataclass
class MeshEstimationSample:
    """
    A data class describing a single mesh estimation sample that comes from a dataset.
    It contains both input image and target information to train a pose estimation model.

    """

    __slots__ = ["image", "vertices_2d", "vertices_3d", "rotation_matrix", "areas", "bboxes_xywh", "is_crowd", "additional_samples"]

    image: Union[np.ndarray, torch.Tensor]
    vertices_2d: Union[np.ndarray, torch.Tensor]
    vertices_3d: Union[np.ndarray, torch.Tensor]
    rotation_matrix: Optional[Union[np.ndarray, torch.Tensor]]
    areas: Optional[np.ndarray]
    bboxes_xywh: Optional[np.ndarray]
    is_crowd: Optional[np.ndarray]
    additional_samples: Optional[List["MeshEstimationSample"]]

    @classmethod
    def compute_area_of_joints_bounding_box(cls, joints) -> np.ndarray:
        """
        Compute area of a bounding box enclosing visible joints for each pose instance.
        :param joints: np.ndarray of [Num Instances, Num Joints, 3] shape (x,y,visibility)
        :return:       np.ndarray of [Num Instances] shape with box area of the visible joints
                       (zero if all joints are not visible or only one joint is visible)
        """
        visible_joints = joints[:, :, 2] > 0
        xmax = np.max(joints[:, :, 0], axis=-1, where=visible_joints, initial=joints[:, :, 0].min())
        xmin = np.min(joints[:, :, 0], axis=-1, where=visible_joints, initial=joints[:, :, 0].max())
        ymax = np.max(joints[:, :, 1], axis=-1, where=visible_joints, initial=joints[:, :, 1].min())
        ymin = np.min(joints[:, :, 1], axis=-1, where=visible_joints, initial=joints[:, :, 1].max())

        w = xmax - xmin
        h = ymax - ymin
        raw_area = w * h
        area = np.clip(raw_area, a_min=0, a_max=None) * (visible_joints.sum(axis=-1, keepdims=False) > 1)
        return area

    def sanitize_sample(self) -> "MeshEstimationSample":
        """
        Apply sanity checks on the pose sample, which includes:
        - Clamp bbox coordinates to ensure they are within image boundaries
        - Update visibility status of keypoints if they are outside of image boundaries
        - Update area if bbox clipping occurs
        This function does not remove instances, but may make them subject for removal instead.
        :return: self
        """
        image_height, image_width, _ = self.image.shape

        # Update joints visibility status
        outside_left = self.vertices_2d[:, :, 0] < 0
        outside_top = self.vertices_2d[:, :, 1] < 0
        outside_right = self.vertices_2d[:, :, 0] >= image_width
        outside_bottom = self.vertices_2d[:, :, 1] >= image_height

        outside_image_mask = outside_left | outside_top | outside_right | outside_bottom
        self.vertices_2d[outside_image_mask, 2] = 0

        if self.bboxes_xywh is not None:
            # Clamp bboxes to image boundaries
            clamped_boxes = xywh_to_xyxy(self.bboxes_xywh, image_shape=(image_height, image_width))
            clamped_boxes = change_bbox_bounds_for_image_size_inplace(clamped_boxes, img_shape=(image_height, image_width))
            clamped_boxes = xyxy_to_xywh(clamped_boxes, image_shape=(image_height, image_width))

            # Recompute sample areas if they are present
            if self.areas is not None:
                area_reduction_factor = clamped_boxes[..., 2:4].prod(axis=-1) / (self.bboxes_xywh[..., 2:4].prod(axis=-1) + 1e-6)
                self.areas = self.areas * area_reduction_factor

            self.bboxes_xywh = clamped_boxes
        return self

    def filter_by_mask(self, mask: np.ndarray) -> "MeshEstimationSample":
        """
        Remove pose instances with respect to given mask.

        :remark: This is main method to modify instances of the sample.
        If you are implementing a subclass of MeshEstimationSample and adding extra field associated with each pose
        instance (Let's say you add a distance property for each pose from the camera), then you should override
        this method to do filtering on extra attribute as well.

        :param mask:   A boolean or integer mask of samples to keep for given sample.
        :return:       A pose sample after filtering.
        """
        self.vertices_2d = self.vertices_2d[mask]
        self.vertices_3d = self.vertices_3d[mask]
        self.rotation_matrix = self.rotation_matrix[mask]
        self.is_crowd = self.is_crowd[mask]
        if self.bboxes_xywh is not None:
            self.bboxes_xywh = self.bboxes_xywh[mask]
        if self.areas is not None:
            self.areas = self.areas[mask]
        return self

    def filter_by_visible_joints(self, min_visible_joints: int) -> "MeshEstimationSample":
        """
        Remove instances from the sample which has less than N visible joints.

        :param min_visible_joints: A minimal number of visible joints a pose has to have in order to be kept.
        :return:                   A pose sample after filtering.
        """
        visible_joints_mask = self.vertices_2d[:, :, 2] > 0
        keep_mask: np.ndarray = np.sum(visible_joints_mask, axis=-1) >= min_visible_joints
        return self.filter_by_mask(keep_mask)

    def filter_by_bbox_area(self, min_bbox_area: Union[int, float]) -> "MeshEstimationSample":
        """
        Remove pose instances that has area of the corresponding bounding box less than a certain threshold.

        :param sample:        Instance of MeshEstimationSample to modify. Modification done in-place.
        :param min_bbox_area: Minimal bounding box area of the pose to keep.
        :return:              A pose sample after filtering.
        """
        if self.bboxes_xywh is None:
            area = self.compute_area_of_joints_bounding_box(self.vertices_2d)
        else:
            area = self.bboxes_xywh[..., 2:4].prod(axis=-1)

        keep_mask = area >= min_bbox_area
        return self.filter_by_mask(keep_mask)

    def filter_by_pose_area(self, min_instance_area: Union[int, float]) -> "MeshEstimationSample":
        """
        Remove pose instances which area is less than a certain threshold.

        :param sample:            Instance of MeshEstimationSample to modify. Modification done in-place.
        :param min_instance_area: Minimal area of the pose to keep.
        :return:                  A pose sample after filtering.
        """

        if self.areas is not None:
            areas = self.areas
        elif self.bboxes_xywh is not None:
            areas = self.bboxes_xywh[..., 2:4].prod(axis=-1, keepdims=False)
        else:
            areas = self.compute_area_of_joints_bounding_box(self.vertices_2d)

        keep_mask = areas >= min_instance_area
        return self.filter_by_mask(keep_mask)
