import os
import json
from fire import Fire
from glob import glob
from tqdm import tqdm


def create_anno_file(folder_path: str):
    annotations = glob(f"{folder_path}/**/metadata/*.json", recursive=True)
    print(f"Found {len(annotations)} annotations")
    results = {}
    for annotation_path in tqdm(annotations):
        _, filename = os.path.split(annotation_path)
        file_id = filename.split(".")[0]
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        results[file_id] = annotation
    with open(f"{folder_path}/annotations.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    Fire(create_anno_file)
