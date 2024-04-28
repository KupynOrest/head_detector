import os

import glob
import cv2
import tqdm
from fire import Fire
from face_retrieval import FaceRetrieval
from pytorch_toolbelt.utils import vstack_header, hstack_autopad


def test_rafd(gen_data_path: str, embeddings_path: str, save_path: str):
    os.makedirs(os.path.join(save_path, "verified"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "not_verified"), exist_ok=True)
    images = glob.glob(f"{gen_data_path}/*")
    retrieval = FaceRetrieval(embeddings_path)

    for index, image_path in tqdm.tqdm(enumerate(images)):
        retrieval_result = retrieval.retrieve(image_path)
        if len(retrieval_result) == 0:
            continue
        retrieval_result = retrieval_result[0]
        if retrieval_result is None:
            continue
        gen_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        ret_image = cv2.cvtColor(cv2.imread(retrieval_result[0].image_path), cv2.COLOR_BGR2RGB)
        print(ret_image.shape)
        ret_image = cv2.resize(ret_image, gen_image.shape[:2][::-1])
        ret_image2 = cv2.cvtColor(cv2.imread(retrieval_result[1].image_path), cv2.COLOR_BGR2RGB)
        ret_image2 = cv2.resize(ret_image2, gen_image.shape[:2][::-1])
        comparison_image = hstack_autopad(
            [
                vstack_header(gen_image, "Generated Image"),
                vstack_header(ret_image, "Retrieved Image"),
                vstack_header(ret_image2, "Retrieved Image 2"),
            ]
        )
        if retrieval_result[0].verified:
            cv2.imwrite(os.path.join(save_path, "verified", f"{index}_{str(retrieval_result[0].distance).replace('.', ',')}.jpg"), cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(os.path.join(save_path, "not_verified", f"{index}_{str(retrieval_result[0].distance).replace('.', ',')}.jpg"), cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    Fire(test_rafd)