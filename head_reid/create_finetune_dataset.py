import os
import json
from typing import Optional

import cv2
from fire import Fire
from glob import glob
from tqdm import tqdm

from data_generator.image_captioning import ImageCaptioner


def create_dataset(image_folder: str, save_folder: str, overfit_identity: Optional[str] = None):
    os.makedirs(save_folder, exist_ok=True)
    captioner = ImageCaptioner()
    dataset = []
    images = glob(os.path.join(image_folder, "*.jpg"))
    images = [x for x in images if "Rafd000_" not in x]
    images = [x for x in images if "Rafd180_" not in x]
    if overfit_identity is not None:
        images = [x for x in images if overfit_identity in x]
    for image_path in tqdm(images):
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        caption = captioner.generate_caption(image)
        _, filename = os.path.split(image_path)
        dataset.append({
            "filename": filename,
            "caption": caption,
        })
        cv2.imwrite(os.path.join(save_folder, filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    with open(os.path.join(save_folder, "annotations.json"), "w") as file:
        json.dump(dataset, file)


if __name__ == "__main__":
    Fire(create_dataset)