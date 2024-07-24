import os

import glob
import cv2
import tqdm
from fire import Fire
from face_retrieval import FaceRetrieval


def test_rafd(gen_data_path: str, embeddings_path: str, save_path: str):
    os.makedirs(os.path.join(save_path, "verified"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "not_verified"), exist_ok=True)
    images = glob.glob(f"{gen_data_path}/*")
    retrieval = FaceRetrieval(embeddings_path, search_k=200000)

    for index, image_path in tqdm.tqdm(enumerate(images)):
        try:
            retrieval_result = retrieval.retrieve(image_path)
            if len(retrieval_result) == 0:
                continue
            retrieval_result = retrieval_result[0]
            if retrieval_result is None:
                continue
            gen_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            ret_image = cv2.cvtColor(cv2.imread(retrieval_result[0].image_path), cv2.COLOR_BGR2RGB)
            if retrieval_result[0].verified:
                print("Verified!")
                cv2.imwrite(os.path.join(save_path, "verified", f"{index}_{str(retrieval_result[0].distance).replace('.', ',')}.jpg"), cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_path, "verified", f"{index}_{str(retrieval_result[0].distance).replace('.', ',')}_retrieved.jpg"), cv2.cvtColor(ret_image, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        #else:
        #    cv2.imwrite(os.path.join(save_path, "not_verified", f"{index}_{str(retrieval_result[0].distance).replace('.', ',')}.jpg"), cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR))
        #    cv2.imwrite(os.path.join(save_path, "not_verified", f"{index}_{str(retrieval_result[0].distance).replace('.', ',')}_retrieved.jpg"), cv2.cvtColor(ret_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    Fire(test_rafd)