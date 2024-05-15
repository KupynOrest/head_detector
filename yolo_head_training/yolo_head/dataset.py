import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

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
from yolo_head.mesh_sample import MeshEstimationSample
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
        mode: Optional[str],
        transforms: List[AbstractKeypointTransform],
        splits: Optional[List[str]] = None,
        crop_bbox_to_visible_keypoints: bool = False,
    ):
        """

        :param data_dir:                     Root directory of the COCO dataset
        :param transforms:                   Transforms to be applied to the image & keypoints

        """
        self.data_dir = data_dir

        if splits is not None:
            images = []
            ann_files = []
            for split in splits:
                split_images, split_anns = self.get_images_and_annotations(os.path.join(data_dir, split))
                images.extend(split_images)
                ann_files.extend(split_anns)
        else:
            images, ann_files = self.get_images_and_annotations(data_dir, mode=mode)

        super().__init__(
            transforms=transforms,
            num_joints=num_joints,
            edge_links=[],
            edge_colors=[],
            keypoint_colors=[(0, 255, 0)] * num_joints,
        )
        self.flame = FLAMELayer(consts=FLAME_CONSTS)
        self.crop_bbox_to_visible_keypoints = crop_bbox_to_visible_keypoints

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

    @classmethod
    def get_images_and_annotations(cls, data_dir: str, mode=None) -> Tuple[List[str], List[str]]:
        data_dir = Path(data_dir)

        images_dir = data_dir / "images"
        excluded_files_list = data_dir / "files.txt"
        if excluded_files_list.exists():
            with excluded_files_list.open("r") as f:
                excluded_files = f.read().splitlines()
        else:
            logger.info(f"Excluded files list not found: {excluded_files_list}")
            excluded_files = []

        images = list(sorted(images_dir.glob("*.jpg")))
        images = [str(x) for x in images if os.path.basename(x) not in excluded_files]
        ann_files = [x.replace("images", "annotations").replace(".jpg", ".npz") for x in images]

        keep_images = []
        keep_anns = []
        for image_file, ann_file in zip(images, ann_files):
            if not os.path.exists(ann_file):
                logger.warning(f"Annotation file not found: {ann_file}")
                continue
            keep_images.append(image_file)
            keep_anns.append(ann_file)

        images = keep_images
        ann_files = keep_anns

        if mode is not None:
            filelist = os.path.join(data_dir, f"{mode}_files.txt")
            with open(filelist, "r") as f:
                files = f.read().splitlines()
            files_to_keep = [os.path.basename(x).split(".")[0] for x in files]

            keep_images = []
            keep_anns = []

            for image_file, ann_file in zip(images, ann_files):
                if os.path.basename(image_file).split(".")[0] not in files_to_keep:
                    continue
                keep_images.append(image_file)
                keep_anns.append(ann_file)

        return keep_images, keep_anns

    def load_sample(self, index: int) -> MeshEstimationSample:
        """
        Read a sample from the disk and return a MeshEstimationSample
        :param index: Sample index
        :return:      Returns an instance of MeshEstimationSample that holds complete sample (image and annotations)
        """
        image_path: str = self.images[index]
        ann_path: str = self.ann_files[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            new_index = random.randint(0, 1000)
            image_path = self.images[new_index]
            ann_path = self.ann_files[new_index]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        head_ann: SampleAnnotation = read_annotation(ann_path, self.flame)

        gt_joints = []
        gt_bboxes_xywh = []
        gt_vertices = []
        gt_rots = []

        for head in head_ann.heads:
            coords = head.get_reprojected_points_in_absolute_coords()
            # gt_joints.append(coords[self.indexes_subset["keypoint_445"]])
            gt_joints.append(coords)
            gt_vertices.append(head.vertices_3d)
            gt_rots.append(head.rotation_matrix)
            gt_bboxes_xywh.append(head.get_face_bbox_xywh())

        gt_bboxes_xywh = np.array(gt_bboxes_xywh).reshape(-1, 4)
        gt_iscrowd = np.zeros(len(gt_joints), dtype=bool).reshape(-1)
        gt_areas = np.prod(gt_bboxes_xywh[:, 2:], axis=1)

        num_instances = len(gt_joints)
        # num_keypoints = len(self.indexes_subset["keypoint_445"])
        gt_joints = np.stack(gt_joints).reshape(num_instances, -1, 2)
        # Add a 1 to the last dimension to get [N, Num Keypoints, 3]
        gt_vertices = np.array(gt_vertices).reshape(num_instances, -1, 3)
        gt_rots = np.array(gt_rots).reshape(num_instances, 3, 3)
        gt_joints = np.concatenate([gt_joints, np.ones((gt_joints.shape[0], gt_joints.shape[1], 1), dtype=gt_joints.dtype)], axis=-1)

        return MeshEstimationSample(
            image=image,
            vertices_2d=gt_joints,
            vertices_3d=gt_vertices,
            rotation_matrix=gt_rots,
            areas=gt_areas,
            bboxes_xywh=gt_bboxes_xywh,
            is_crowd=gt_iscrowd,
            additional_samples=None,
        )

    def __getitem__(self, index: int) -> MeshEstimationSample:
        sample = self.load_sample(index)
        sample = self.transforms.apply_to_sample(sample)

        # Update bounding boxes and areas to match the visible joints area
        if self.crop_bbox_to_visible_keypoints:
            if len(sample.joints):
                visible_joints = sample.joints[:, :, 2] > 0
                xmax = np.max(sample.joints[:, :, 0], axis=-1, where=visible_joints, initial=sample.joints[:, :, 0].min())
                xmin = np.min(sample.joints[:, :, 0], axis=-1, where=visible_joints, initial=sample.joints[:, :, 0].max())
                ymax = np.max(sample.joints[:, :, 1], axis=-1, where=visible_joints, initial=sample.joints[:, :, 1].min())
                ymin = np.min(sample.joints[:, :, 1], axis=-1, where=visible_joints, initial=sample.joints[:, :, 1].max())

                w = xmax - xmin
                h = ymax - ymin
                raw_area = w * h
                area = np.clip(raw_area, a_min=0, a_max=None) * (visible_joints.sum(axis=-1, keepdims=False) > 1)
                sample.bboxes_xywh = np.stack([xmin, ymin, w, h], axis=1)
                sample.areas = area

        return sample

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
