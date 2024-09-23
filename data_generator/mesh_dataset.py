import os
import abc
import glob
import shutil
from os import environ
from typing import Tuple, Union, Optional

import cv2
import numpy as np
from fire import Fire
from tqdm import tqdm
from pycocotools.coco import COCO

from dad_3d_heads.predictor import FaceMeshPredictor
from binary_detector import HeadDetector


class HeadInfo:
    def __init__(self, x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose):
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        self.blur = blur
        self.expression = expression
        self.illumination = illumination
        self.invalid = invalid
        self.occlusion = occlusion
        self.pose = pose

    def __repr__(self):
        return (f"HeadInfo(x1={self.x1}, y1={self.y1}, w={self.w}, h={self.h}, blur={self.blur}, "
                f"expression={self.expression}, illumination={self.illumination}, invalid={self.invalid}, "
                f"occlusion={self.occlusion}, pose={self.pose})")


class MeshDatasetCreator:
    def __init__(self, image_folder: str, save_path: str):
        self.predictor = FaceMeshPredictor.dad_3dnet()
        self.save_path = save_path
        self.image_folder = image_folder

    @abc.abstractmethod
    def _get_list_of_items(self):
        pass

    @abc.abstractmethod
    def _get_image(self, item) -> Tuple[np.ndarray, str]:
        pass

    @abc.abstractmethod
    def _get_bboxes(self, item, image):
        pass

    def process_dataset(self):
        os.makedirs(os.path.join(self.save_path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, "annotations"), exist_ok=True)
        for item in tqdm(self._get_list_of_items()):
            image, filename = self._get_image(item=item)
            bboxes = self._get_bboxes(item=item, image=image)
            mesh_annotations = []
            for bbox in bboxes:
                try:
                    x, y, w, h = ensure_bbox_boundaries(extend_bbox(np.array(bbox), 0.25), image.shape[:2])
                    cropped_img = image[y: y + h, x: x + w]
                    result = self.predictor(cropped_img)
                    mesh_annotations.append({
                        "bbox": bbox,
                        "extended_bbox": [int(x), int(y), int(w), int(h)],
                        "3dmm_params": result["3dmm_params"].cpu().numpy()
                    })
                except:
                    pass
            if len(mesh_annotations) > 0:
                stacked_annotations = {key: np.stack([d[key] for d in mesh_annotations]) for key in
                                       mesh_annotations[0].keys()}
                np.savez(os.path.join(self.save_path, "annotations", filename.replace("jpg", "npz")),
                         **stacked_annotations)
            if not os.path.isfile(os.path.join(self.save_path, "images", filename)):
                cv2.imwrite(os.path.join(self.save_path, "images", filename),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class MeshDatasetFromCOCO(MeshDatasetCreator):
    def __init__(self, image_folder: str, save_path: str, coco_path: str):
        super().__init__(image_folder=image_folder, save_path=save_path)
        self.coco = COCO(coco_path)
        self.coco_path = coco_path

    def _get_list_of_items(self):
        img_ids = self.coco.getImgIds()
        return img_ids

    def _get_image(self, item) -> Tuple[np.ndarray, str]:
        image_info = self.coco.loadImgs(item)[0]
        image_path = os.path.join(self.image_folder, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, os.path.basename(image_path)

    def _get_bboxes(self, item, image):
        ann_ids = self.coco.getAnnIds(imgIds=item)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = [x['bbox'] for x in anns]
        return bboxes

    def process_dataset(self):
        super().process_dataset()
        shutil.copy(self.coco_path, os.path.join(self.save_path, "coco_annotations.json"))
        shutil.copy(self.coco_path.replace("coco.json", "train_annotations.json"),
                    os.path.join(self.save_path, "train_annotations.json"))
        shutil.copy(self.coco_path.replace("coco.json", "val_annotations.json"),
                    os.path.join(self.save_path, "val_annotations.json"))


class MeshDatasetFromWIDER(MeshDatasetCreator):
    def __init__(self, image_folder: str, save_path: str, wider_path: str):
        super().__init__(image_folder=image_folder, save_path=save_path)
        self.data = self._load_wider(wider_path)

    @staticmethod
    def _load_wider(file_path: str):
        annotations = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
        i = 0
        while i < len(lines):
            filename = lines[i].strip()
            num_boxes = int(lines[i + 1].strip())
            head_infos = []
            for j in range(num_boxes):
                bbox_data = list(map(int, lines[i+2+j].strip().split()))
                head_info = HeadInfo(*bbox_data)
                head_infos.append(head_info)

            annotations[filename] = head_infos
            if num_boxes == 0:
                i += 3
            else:
                i += 2 + num_boxes
        return annotations

    def _get_list_of_items(self):
        items = self.data.items()
        return items

    def _get_image(self, item) -> Tuple[np.ndarray, str]:
        image_name, head_infos = item
        image_path = os.path.join(self.image_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, os.path.basename(image_path)

    def _get_bboxes(self, item, image):
        image_name, head_infos = item
        bboxes = [[head.x1, head.y1, head.w, head.h] for head in head_infos]
        return bboxes


class MeshDatasetFromImages(MeshDatasetCreator):
    def __init__(self, image_folder: str, save_path: str, model_path: str):
        super().__init__(image_folder=image_folder, save_path=save_path)
        self.detector = HeadDetector(model_path)

    def _get_list_of_items(self):
        files = glob.glob(f"{self.image_folder}/images/*.jpg")
        with open(f"{self.image_folder}/files.txt", 'r') as file:
            filtered_files = [line.strip() for line in file]
        files = [x for x in files if os.path.basename(x) not in filtered_files]
        return files

    def _get_image(self, item) -> Tuple[np.ndarray, str]:
        image_path = item
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, os.path.basename(image_path)

    def _get_bboxes(self, item, image):
        bboxes = self.detector(image=image)
        return np.array([x.to_xywh() for x in bboxes])


def extend_bbox(bbox: np.array, offset: Union[Tuple[float, ...], float] = 0.1) -> np.array:
    """
    Increases bbox dimensions by offset*100 percent on each side.

    IMPORTANT: Should be used with ensure_bbox_boundaries, as might return negative coordinates for x_new, y_new,
    as well as w_new, h_new that are greater than the image size the bbox is extracted from.

    :param bbox: [x, y, w, h]
    :param offset: (left, right, top, bottom), or (width_offset, height_offset), or just single offset that specifies
    fraction of spatial dimensions of bbox it is increased by.

    For example, if bbox is a square 100x100 pixels, and offset is 0.1, it means that the bbox will be increased by
    0.1*100 = 10 pixels on each side, yielding 120x120 bbox.

    :return: extended bbox, [x_new, y_new, w_new, h_new]
    """
    x, y, w, h = bbox

    if isinstance(offset, tuple):
        if len(offset) == 4:
            left, right, top, bottom = offset
        elif len(offset) == 2:
            w_offset, h_offset = offset
            left = right = w_offset
            top = bottom = h_offset
    else:
        left = right = top = bottom = offset

    return np.array([x - w * left, y - h * top, w * (1.0 + right + left), h * (1.0 + top + bottom)]).astype("int32")


def ensure_bbox_boundaries(bbox: np.array, img_shape: Tuple[int, int]) -> np.array:
    """
    Trims the bbox not the exceed the image.
    :param bbox: [x, y, w, h]
    :param img_shape: (h, w)
    :return: trimmed to the image shape bbox
    """
    x1, y1, w, h = bbox
    x1, y1 = min(max(0, x1), img_shape[1]), min(max(0, y1), img_shape[0])
    x2, y2 = min(max(0, x1 + w), img_shape[1]), min(max(0, y1 + h), img_shape[0])
    w, h = x2 - x1, y2 - y1
    return np.array([x1, y1, w, h]).astype("int32")


def get_folder_name() -> str:
    if "SLURM_ARRAY_TASK_ID" not in environ:
        return "split_00000"
    task_id = int(environ["SLURM_ARRAY_TASK_ID"])
    return f"split_{task_id:05d}"


def create_dataset(image_folder: str, save_path: str, from_coco: bool = True, detector_path: Optional[str] = None, coco_path: Optional[str] = None):
    if from_coco:
        annotations_generator = MeshDatasetFromCOCO(image_folder=image_folder, save_path=save_path, coco_path=coco_path)
    else:
        folder_name = get_folder_name()
        image_folder = os.path.join(image_folder, folder_name)
        annotations_generator = MeshDatasetFromImages(image_folder=image_folder, save_path=os.path.join(save_path, folder_name), model_path=detector_path)
    annotations_generator.process_dataset()


if __name__ == '__main__':
    Fire(create_dataset)
