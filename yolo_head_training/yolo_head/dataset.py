import os
from typing import List

import cv2
import numpy as np
from pytorch_toolbelt.utils import fs
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.object_names import Processings
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import (
    AbstractPoseEstimationDataset,
)
from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform
from tqdm import tqdm

from yolo_head.dataset_parsing import read_annotation, SampleAnnotation
from yolo_head.flame import get_indices, FLAMELayer, FLAME_CONSTS

logger = get_logger(__name__)


@register_dataset()
class DAD3DHeadsDataset(AbstractPoseEstimationDataset):
    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        data_dir: str,
        num_joints: int,
        mode: str,
        transforms: List[AbstractKeypointTransform],
    ):
        """

        :param data_dir:                     Root directory of the COCO dataset
        :param transforms:                   Transforms to be applied to the image & keypoints

        """
        self.data_dir = data_dir
        images = fs.find_images_in_dir(os.path.join(data_dir, "images"))
        images = self.filter_images(images, mode)
        ann_files = [x.replace("images", "annotations").replace(".jpg", ".npz") for x in images]

        for ann_file in ann_files:
            if not os.path.exists(ann_file):
                raise ValueError(f"Annotation file not found: {ann_file}")

        super().__init__(
            transforms=transforms,
            num_joints=num_joints,
            edge_links=[],
            edge_colors=[],
            keypoint_colors=[(0, 255, 0)] * num_joints,
        )
        self.flame = FLAMELayer(consts=FLAME_CONSTS)

        if False:
            # A check to keep only large boxes
            keep_images = []
            keep_anns = []

            for image, ann_file in zip(tqdm(images), ann_files):
                ann = read_annotation(ann_file, self.flame)
                image_h, image_w = cv2.imread(image).shape[:2]
                scale = 640 / max(image_h, image_w)

                if len(ann.heads) == 0:
                    continue

                # Filter all images where minimal head is smaller than 192x192 in 640x640 image
                min_head_area = min([head.get_face_bbox_area() * scale * scale for head in ann.heads])
                if min_head_area < 192 * 192:
                    continue
                keep_images.append(image)
                keep_anns.append(ann_file)

            images = keep_images
            ann_files = keep_anns

        self.images = np.array(images)
        self.ann_files = np.array(ann_files)
        self.indexes_subset = get_indices()

    def __len__(self):
        return len(self.images)

    def filter_images(self, image_files: List[str], mode: str):
        filelist = os.path.join(self.data_dir, f"{mode}_files.txt")
        if os.path.exists(filelist):
            with open(filelist, "r") as f:
                files = f.read().splitlines()
            files = [x.split(".")[0] for x in files]
            image_files = [x for x in image_files if os.path.basename(x).split(".")[0] in files]
        return image_files

    def load_sample(self, index: int) -> PoseEstimationSample:
        """
        Read a sample from the disk and return a PoseEstimationSample
        :param index: Sample index
        :return:      Returns an instance of PoseEstimationSample that holds complete sample (image and annotations)
        """
        image_path: str = self.images[index]
        ann_path: str = self.ann_files[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        head_ann: SampleAnnotation = read_annotation(ann_path, self.flame)

        gt_joints = []
        gt_bboxes_xywh = []

        for head in head_ann.heads:
            coords = head.get_reprojected_points_in_absolute_coords()
            gt_joints.append(coords[self.indexes_subset["keypoint_445"]])
            gt_bboxes_xywh.append(head.get_face_bbox_xywh())

        gt_bboxes_xywh = np.array(gt_bboxes_xywh)
        gt_iscrowd = np.zeros(len(gt_joints), dtype=bool)
        gt_areas = np.prod(gt_bboxes_xywh[:, 2:], axis=1)

        gt_joints = np.stack(gt_joints)
        # Add a 1 to the last dimension to get [N, Num Keypoints, 3]
        gt_joints = np.concatenate([gt_joints, np.ones((gt_joints.shape[0], gt_joints.shape[1], 1), dtype=gt_joints.dtype)], axis=-1)

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
        )
        return params


__all__ = ("DAD3DHeadsDataset",)
