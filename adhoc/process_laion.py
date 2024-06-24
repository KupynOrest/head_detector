import json
from typing import Optional, Dict, Any
import uuid

import cv2
import tqdm
import glob
import os
from fire import Fire
import numpy as np

from data_generator.yolo_pose_processor import PoseProcessor
from data_generator.image_captioning import ImageCaptioner

MAX_ASPECT_RATIO = 2


def generate_unique_filename(prefix: Optional[str] = None) -> str:
    unique_filename = uuid.uuid4().hex
    if prefix is not None:
        unique_filename = prefix + '_' + unique_filename
    return unique_filename


def valid_image(image: np.ndarray, metadata: Dict[str, Any]) -> bool:
    if image.shape[0] / image.shape[1] > MAX_ASPECT_RATIO or image.shape[1] / image.shape[0] > MAX_ASPECT_RATIO:
        return False
    if metadata["NSFW"] == "NSFW" or metadata["NSFW"] == "UNSURE":
        return False
    return True


def dataset(laion_path: str, save_dir: str, split: str = "split_00000"):
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "metadata"), exist_ok=True)
    pose_processor = PoseProcessor()
    image_captioner = ImageCaptioner("blip2-2.7b")
    images = glob.glob(f"{laion_path}/{split}/**/*.jpg", recursive=True)
    print(f"Found {len(images)} images")
    for index, image_path in enumerate(tqdm.tqdm(images)):
        try:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            with open(image_path.replace(".jpg", ".json"), "r") as file:
                metadata = json.load(file)
            if not valid_image(image, metadata):
                continue
            caption = image_captioner.generate_caption(image)
            pose_image = pose_processor(image)
            filename = generate_unique_filename(prefix="laion")
            pose_image.save(os.path.join(save_dir, "images", f"{filename}.jpg"))
            generated_metadata = {
                "caption": caption,
                "original_caption": metadata["caption"],
            }
            with open(os.path.join(save_dir, "metadata", f"{filename}.json"), "w") as file:
                json.dump(generated_metadata, file)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    Fire(dataset)
