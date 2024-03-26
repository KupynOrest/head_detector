import os
from typing import Tuple, List, Union

import cv2
import numpy as np
from pytorch_toolbelt.utils import fs
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.object_names import Processings
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xyxy_to_xywh
from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import (
    AbstractPoseEstimationDataset,
)
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform

from yolo_head_training.yolo_head.dataset_parsing import read_annotation, SampleAnnotation

logger = get_logger(__name__)


@register_dataset()
class DAD3DHeadsDataset(AbstractPoseEstimationDataset):
    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        data_dir: str,
        num_joints: int,
        transforms: List[AbstractKeypointTransform],
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
    ):
        """

        :param data_dir:                     Root directory of the COCO dataset
        :param transforms:                   Transforms to be applied to the image & keypoints
        :param edge_links:                   Edge links between joints
        :param edge_colors:                  Color of the edge links. If None, the color will be generated randomly.
        :param keypoint_colors:              Color of the keypoints. If None, the color will be generated randomly.

        """
        images = fs.find_images_in_dir(os.path.join(data_dir, "images"))
        ann_files = [os.path.join(data_dir, "annotations", fs.id_from_fname(x) + ".json") for x in images]

        for ann_file in ann_files:
            if not os.path.exists(ann_file):
                raise ValueError("")

        super().__init__(
            transforms=transforms,
            num_joints=num_joints,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )
        self.images = np.array(images)
        self.ann_files = np.array(ann_files)

    def __len__(self):
        return len(self.images)

    def load_sample(self, index: int) -> PoseEstimationSample:
        """
        Read a sample from the disk and return a PoseEstimationSample
        :param index: Sample index
        :return:      Returns an instance of PoseEstimationSample that holds complete sample (image and annotations)
        """
        image_path: str = self.images[index]
        ann_path: str = self.ann_files[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_height, image_width = image.shape[:2]

        head_ann: SampleAnnotation = read_annotation(ann_path)

        gt_joints = []
        gt_bboxes_xywh = []

        for head in head_ann.heads:
            gt_joints.append(head.get_points_in_absolute_coords())
            gt_bboxes_xywh.append(head.get_face_bbox_xywh())

        gt_bboxes_xywh = np.array(gt_bboxes_xywh)
        gt_iscrowd = np.zeros(len(gt_joints), dtype=bool)
        gt_areas = np.prod(gt_bboxes_xywh[:, 2:], axis=1)

        gt_joints = np.array(gt_joints).reshape(-1, 68, 2)
        # Add a 1 to the last dimension to get [N, 68, 3]
        gt_joints = np.concatenate(
            [gt_joints, np.ones((gt_joints.shape[0], gt_joints.shape[1], 1), dtype=gt_joints.dtype)], axis=-1
        )

        return PoseEstimationSample(
            image=image,
            mask=np.ones(image.shape[:2], dtype=np.float32),
            joints=gt_joints,
            areas=gt_areas,
            bboxes_xywh=gt_bboxes_xywh,
            is_crowd=gt_iscrowd,
            additional_samples=None,
        )

    def get_dataset_preprocessing_params(self) -> dict:
        """
        This method returns a dictionary of parameters describing preprocessing steps to be applied to the dataset.
        :return:
        """
        rgb_to_bgr = {Processings.ReverseImageChannels: {}}
        image_to_tensor = {Processings.ImagePermute: {"permutation": (2, 0, 1)}}
        pipeline = [rgb_to_bgr] + self.transforms.get_equivalent_preprocessing() + [image_to_tensor]
        params = dict(
            conf=0.5,
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
        )
        return params


__all__ = ("DAD3DHeadsDataset",)
