import os
from os import environ
import json
import uuid
import random
from typing import Optional, List, Tuple

import cv2
import tqdm
import glob
from PIL import Image
from fire import Fire
import numpy as np

from data_generator.generation_pipeline import get_pipeline
from data_generator.safety_checker import UnsafeContentDetector
from data_generator.caption_processor import CaptionProcessor


NEGATIVE_PROMPT = "worst quality, low quality, sketch, bad anatomy, bad body, bad face, nsfw, nudity, violence"
MAX_ASPECT_RATIO = 2
MAX_TASKS = 100


class DataGenerator:
    def __init__(self):
        self.pipeline = get_pipeline()
        self.caption_processor = CaptionProcessor()
        self.safety_checker = UnsafeContentDetector()

    @staticmethod
    def valid_image(image: np.ndarray) -> bool:
        if image.shape[0] / image.shape[1] > MAX_ASPECT_RATIO or image.shape[1] / image.shape[0] > MAX_ASPECT_RATIO:
            return False
        return True

    @staticmethod
    def generate_unique_filename(prefix: Optional[str] = None) -> str:
        unique_filename = uuid.uuid4().hex
        if prefix is not None:
            unique_filename = prefix + '_' + unique_filename
        return unique_filename

    def _get_start_end_index(self, images: List[str]) -> Tuple[int, int]:
        if "SLURM_ARRAY_TASK_ID" not in environ:
            return 0, len(images)
        task_id = int(environ["SLURM_ARRAY_TASK_ID"])
        num_in_one_bucket = len(images) // MAX_TASKS
        return task_id * num_in_one_bucket, min(len(images), (task_id + 1) * num_in_one_bucket)

    def _get_folder_name(self) -> str:
        if "SLURM_ARRAY_TASK_ID" not in environ:
            return "split_00000"
        task_id = int(environ["SLURM_ARRAY_TASK_ID"])
        return f"split_{task_id:05d}"

    def generate(self, data_path: str, save_dir: str):
        folder_name = self._get_folder_name()
        with open(os.path.join(data_path, "annotations.json"), "r") as file:
            annotations = json.load(file)
        os.makedirs(os.path.join(save_dir, folder_name, "images"), exist_ok=True)
        pose_images = glob.glob(f"{data_path}/**/images/*.jpg", recursive=True)
        print(f"Found {len(pose_images)} images")
        start, end = self._get_start_end_index(pose_images)
        print(f"Reading subset from {start} to {end}, total: {len(pose_images)}")

        for index in tqdm.tqdm(range(start, end)):
            try:
                image_path = pose_images[index]
                filename = os.path.split(image_path)[1].split('.')[0]
                if os.path.exists(os.path.join(save_dir, folder_name, "images", f"{filename}.jpg")):
                    continue
                pose_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                if not self.valid_image(pose_image):
                    continue
                caption = annotations[filename]["caption"]
                if self.caption_processor.contains_person(caption):
                    continue
                caption = self.caption_processor.add_ethnic_labels(caption)
                if random.random() < 0.7:
                    caption = f"{caption}, ultra highres, hyper realistic and highly detailed, detailed face, sharp eyes, realistic skin texture"
                image = self.pipeline(
                    caption,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=Image.fromarray(pose_image),
                    num_inference_steps=40,
                    adapter_conditioning_scale=0.8,
                    guidance_scale=7.0,
                ).images[0]
                if self.safety_checker(np.array(image)):
                    continue
                image.save(os.path.join(save_dir, folder_name, "images", f"{filename}.jpg"))
            except:
                pass


def dataset(data_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    generator = DataGenerator()
    generator.generate(data_path, save_dir)


if __name__ == "__main__":
    Fire(dataset)
