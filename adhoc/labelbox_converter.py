import os
import ujson as json
from typing import Dict, Any, List

from fire import Fire
from tqdm import tqdm

import json
import random
from pycocotools.coco import COCO

TRAIN_RATIO = 0.8


def split_annotations(input_json, train_ratio=0.8):
    # Initialize COCO object
    coco = COCO(input_json)

    # Get image IDs
    image_ids = coco.getImgIds()

    # Shuffle image IDs
    random.shuffle(image_ids)

    # Split image IDs into train and val sets
    num_train = int(len(image_ids) * train_ratio)
    train_image_ids = image_ids[:num_train]
    val_image_ids = image_ids[num_train:]

    # Get corresponding annotations for train and val sets
    train_annotations = coco.loadAnns(coco.getAnnIds(imgIds=train_image_ids))
    val_annotations = coco.loadAnns(coco.getAnnIds(imgIds=val_image_ids))

    # Create COCO instances for train and val sets
    train_coco = {
        'info': coco.dataset['info'],
        'licenses': coco.dataset['licenses'],
        'images': coco.loadImgs(train_image_ids),
        'annotations': train_annotations,
        'categories': coco.dataset['categories']
    }

    val_coco = {
        'info': coco.dataset['info'],
        'licenses': coco.dataset['licenses'],
        'images': coco.loadImgs(val_image_ids),
        'annotations': val_annotations,
        'categories': coco.dataset['categories']
    }
    return train_coco, val_coco


def save_annotations(coco_instance, output_json):
    with open(output_json, 'w') as f:
        json.dump(coco_instance, f)


class Labelbox2COCOConverter:
    @staticmethod
    def _read_data(annotation_path: str) -> List[Dict[str, Any]]:
        data = list(map(json.loads, open(annotation_path)))
        return data

    @staticmethod
    def _should_skip(annotation: Dict[str, Any]) -> bool:
        for project_id, project_data in annotation['projects'].items():
            labels = project_data['labels']
            if len(labels) == 0:
                return True
            for label in labels:
                if len(label['annotations']['classifications']) > 0:
                    label_names = [x['radio_answer']['name'] for x in label['annotations']['classifications']]
                    for x in label_names:
                        if x == 'No_Head_Visible':
                            return False
                if len(label['annotations']['objects']) == 0:
                    return True
        return False

    def __call__(self, annotation_path: str) -> Dict[str, Any]:
        coco_format = {
            "info": {},
            "licenses": [],
            "categories": [{"id": 1, "name": "head"}],
            "images": [],
            "annotations": []
        }
        annotation_id = 1
        image_id = 1
        for annotation in tqdm(self._read_data(annotation_path)):
            if self._should_skip(annotation):
                continue
            data_row = annotation['data_row']
            image_info = {
                "id": image_id,
                "file_name": data_row['external_id'],
                "height": annotation['media_attributes']['height'],
                "width": annotation['media_attributes']['width'],
                "license": None,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": None
            }
            coco_format["images"].append(image_info)
            for project_id, project_data in annotation['projects'].items():
                labels = project_data['labels']
                for label in labels:
                    for obj in label['annotations']['objects']:
                        bbox = obj['bounding_box']
                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "segmentation": [],
                            "area": int(bbox['height'] * bbox['width']),
                            "bbox": list(map(int, [bbox['left'], bbox['top'], bbox['width'], bbox['height']])),
                            "iscrowd": 0
                        }
                        coco_format["annotations"].append(annotation_info)
                        annotation_id += 1
            image_id += 1
        return coco_format


def convert(labelbox_json_path: str, save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)
    converter = Labelbox2COCOConverter()
    data = converter(labelbox_json_path)
    with open(os.path.join(save_path, "coco.json"), "w") as fd:
        json.dump(data, fd, indent=2, ensure_ascii=False)
    train_annotations, val_annotations = split_annotations(os.path.join(save_path, "coco.json"), TRAIN_RATIO)

    train_output_json = 'train_annotations.json'  # Path to save the train annotations file
    val_output_json = 'val_annotations.json'  # Path to save the validation annotations file

    save_annotations(train_annotations, os.path.join(save_path, train_output_json))
    save_annotations(val_annotations, os.path.join(save_path, val_output_json))


if __name__ == "__main__":
    Fire(convert)
