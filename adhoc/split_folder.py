import os
from fire import Fire
from glob import glob
from tqdm import tqdm


def split_folder(folder_path: str, num_folders: int = 100):
    images = glob(f"{folder_path}/images/*.jpg", recursive=True)
    print(f"Found {len(images)} images")
    num_images_per_folder = len(images) // num_folders
    for index, image_path in tqdm(enumerate(images)):
        folder_index = index // num_images_per_folder
        folder_name = f"split_{str(folder_index).zfill(5)}"
        os.makedirs(os.path.join(folder_path, folder_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(folder_path, folder_name, "metadata"), exist_ok=True)
        os.rename(image_path, os.path.join(folder_path, folder_name, "images", os.path.basename(image_path)))
        metadata_path = image_path.replace("images", "metadata").replace(".jpg", ".json")
        os.rename(metadata_path, os.path.join(folder_path, folder_name, "metadata", os.path.basename(metadata_path)))


if __name__ == "__main__":
    Fire(split_folder)
