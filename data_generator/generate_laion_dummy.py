import os
from os import environ
import json
import uuid
from typing import Optional, List, Tuple

import cv2
import tqdm
import glob
from PIL import Image
from fire import Fire
import numpy as np

from data_generator.generation_pipeline import get_pipeline


NEGATIVE_PROMPT = "worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, bad anatomy, bad body, bad face, nsfw, nudity, violence"
MAX_ASPECT_RATIO = 2
MAX_TASKS = 100


class DataGenerator:
    def __init__(self):
        self.pipeline = get_pipeline()

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

    def generate(self, data_path: str, save_dir: str):
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        pose_images = glob.glob(f"{data_path}/images/*.jpg", recursive=True)
        print(f"Found {len(pose_images)} images")

        for image_path in tqdm.tqdm(pose_images[:1500]):
            filename = os.path.split(image_path)[1].split('.')[0]
            pose_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            pose_image = cv2.resize(pose_image, (pose_image.shape[1] // 2, pose_image.shape[0] // 2))
            with open(image_path.replace("images", "metadata").replace("jpg", "json"), "r") as file:
                annotations = json.load(file)
            caption = annotations["caption"]
            image = self.pipeline(
                f"{caption}, ultra highres, hyper realistic and highly detailed, detailed face, sharp eyes, realistic skin texture",
                negative_prompt=NEGATIVE_PROMPT,
                image=pose_image,
                num_inference_steps=40,
                adapter_conditioning_scale=0.9,
                guidance_scale=7.0,
            ).images[0]
            image.save(os.path.join(save_dir, "images", f"{filename}.jpg"))


def dataset(data_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    generator = DataGenerator()
    generator.generate(data_path, save_dir)


if __name__ == "__main__":
    Fire(dataset)
