import os
import json

from annoy import AnnoyIndex
from glob import glob
from tqdm import tqdm
from fire import Fire
from deepface import DeepFace
from multiprocessing import Pool, Manager


EMBEDDING_SIZE = 4096


def process_images(image_paths, index_queue):
    embeddings = []
    path_mapping = {}
    for image_path in tqdm(image_paths):
        try:
            embedding_objs = DeepFace.represent(img_path=image_path, model_name="DeepFace", enforce_detection=False)
            for embedding_obj in embedding_objs:
                if embedding_obj["face_confidence"] < 0.4:
                    continue
                index = index_queue.get()
                embeddings.append({'index': index, 'embedding': embedding_obj["embedding"]})
                path_mapping[index] = image_path
        except Exception as e:
            print(e)
            pass
    return embeddings, path_mapping


def build_face_index(image_folder: str, save_path: str, num_trees: int = 10, num_jobs: int = 8):
    os.makedirs(save_path, exist_ok=True)
    images = glob(f"{image_folder}/*.jpg", recursive=True)

    # Split images into chunks for each process
    chunk_size = len(images) // num_jobs
    image_chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]

    # Use Manager to handle shared data
    manager = Manager()
    index_queue = manager.Queue()

    # Fill the queue with indices
    for i in range(len(images)):
        index_queue.put(i)

    with Pool(processes=num_jobs) as pool:
        results = pool.starmap(process_images, [(chunk, index_queue) for chunk in image_chunks])

    # Merge results
    t = AnnoyIndex(EMBEDDING_SIZE, 'angular')
    path_mapping = {}
    for embeddings, mapping in results:
        for item in embeddings:
            t.add_item(item['index'], item['embedding'])
        path_mapping.update(mapping)

    t.build(num_trees)
    t.save(os.path.join(save_path, "face_index.ann"))

    with open(os.path.join(save_path, "path_mapping.json"), "w") as file:
        json.dump(path_mapping, file)


if __name__ == "__main__":
    Fire(build_face_index)
