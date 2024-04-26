import numpy as np
import torch
import cv2
from fire import Fire
from glob import glob
from tqdm import tqdm
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


def array_to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.transpose((2, 0, 1))).unsqueeze(0)


def compute_image_quality(folder: str, laion_path: str):
    inception = InceptionScore()
    fid = FrechetInceptionDistance(feature=64)
    files = list(sorted(glob(f"{folder}/*")))

    real_images = glob(f"{laion_path}/split_00000/00000/*.jpg", recursive=True)
    for real_image in tqdm(real_images[:1500]):
        real_image = cv2.cvtColor(cv2.imread(real_image), cv2.COLOR_BGR2RGB)
        fid.update(array_to_tensor(real_image), real=True)

    for file in tqdm(files):
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        img_tensor = array_to_tensor(image)
        inception.update(img_tensor)
        fid.update(img_tensor, real=False)
    print(f'Inception score: {inception.compute()}')
    print(f'FID score: {fid.compute()}')


if __name__ == "__main__":
    Fire(compute_image_quality)
