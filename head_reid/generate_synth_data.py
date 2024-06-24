import os
import json

import torch
from fire import Fire
from tqdm import tqdm
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler


def get_pipeline(model_path: str) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    return pipe


def generate_data(annotations_dir: str, save_dir: str, model_path: str = "/home/okupyn/head_detector/head_reid/rafd-model") -> None:
    pipe = get_pipeline(model_path)
    with open(f"{annotations_dir}/annotations.json", "r") as file:
        annotations = json.load(file)
    os.makedirs(save_dir, exist_ok=True)

    for index, annotation in tqdm(enumerate(annotations)):
        caption = annotation["caption"]
        image = pipe(f'{caption}',  width=768, height=1152, num_inference_steps=50, guidance_scale=7.5).images[0]
        image.save(f"{save_dir}/{index}.jpg")


if __name__ == "__main__":
    Fire(generate_data)
