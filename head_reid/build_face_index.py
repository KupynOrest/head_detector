import os
import json

from annoy import AnnoyIndex
from glob import glob
from tqdm import tqdm
import cv2
from fire import Fire
from deepface import DeepFace


EMBEDDING_SIZE = 4096


def build_face_index(image_folder: str, save_path: str, num_trees: int = 10):
    os.makedirs(save_path, exist_ok=True)
    t = AnnoyIndex(EMBEDDING_SIZE, 'angular')
    path_mapping = {}
    index = 0
    images = glob(f"{image_folder}/*.jpg", recursive=True)
    #images = glob(f"{image_folder}/*.jpg", recursive=True)
    for image_path in tqdm(images):
        try:
            embedding_objs = DeepFace.represent(img_path=image_path, model_name="DeepFace", enforce_detection=False)
            for embedding_obj in embedding_objs:
                if embedding_obj["face_confidence"] < 0.4:
                    continue
                t.add_item(index, embedding_obj["embedding"])
                path_mapping[index] = image_path
                index += 1
        except Exception as e:
            print(e)
            pass
    t.build(num_trees)
    t.save(os.path.join(save_path, "face_index.ann"))
    with open(os.path.join(save_path, "path_mapping.json"), "w") as file:
        json.dump(path_mapping, file)


if __name__ == "__main__":
    Fire(build_face_index)
