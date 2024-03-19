import os
import json
import uuid
from typing import Optional

import cv2
import tqdm
import glob
from fire import Fire
import numpy as np

from data_generator.generation_pipeline import get_pipeline
from data_generator.safety_checker import UnsafeContentDetector


NEGATIVE_PROMPT = "worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch, bad anatomy, bad body, bad face, nsfw, nudity, violence"
MAX_ASPECT_RATIO = 2


class DataGenerator:
    def __init__(self):
        self.unsafe_content_detector = UnsafeContentDetector()
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

    def generate(self, data_path: str, save_dir: str, max_images: int = 40000):
        pose_images = glob.glob(f"{data_path}/images/*.jpg", recursive=True)
        print(f"Found {len(pose_images)} images")
        for index, image_path in enumerate(tqdm.tqdm(pose_images)):
            try:
                pose_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                if not self.valid_image(pose_image):
                    continue
                with open(image_path.replace("images", "metadata").replace(".jpg", ".json"), "r") as file:
                    metadata = json.load(file)
                caption = metadata["caption"]
                image = self.pipeline(
                    f"{caption}, realistic",
                    negative_prompt=NEGATIVE_PROMPT,
                    image=pose_image,
                    num_inference_steps=30,
                    adapter_conditioning_scale=0.8,
                    guidance_scale=7.0,
                ).images[0]
                if self.unsafe_content_detector(np.array(image)):
                    continue
                filename = self.generate_unique_filename(prefix="laion")
                image.save(os.path.join(save_dir, f"{filename}.jpg"))
            except Exception as e:
                pass


def dataset(data_path: str, save_dir: str, max_images: int = 40000):
    os.makedirs(save_dir, exist_ok=True)
    generator = DataGenerator()
    generator.generate(data_path, save_dir, max_images)


if __name__ == "__main__":
    Fire(dataset)
