import os
import cv2
import glob
from insightface.app import FaceAnalysis
from fire import Fire
import tqdm
from insightface.utils import face_align


def align(pattern: str):
    os.makedirs("aligned_heads/if", exist_ok=True)
    # Load image
    images = glob.glob(pattern)
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(224, 224))
    for index, image_path in enumerate(tqdm.tqdm(images)):
        image = cv2.imread(image_path)
        input_size = 256
        faces = app.get(image, max_num=1)
        if len(faces) != 1:
            continue
        bbox = faces[0].bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = input_size / (max(w, h) * 1.5)
        aimg, _ = face_align.transform(image, center, input_size, _scale, rotate)
        cv2.imwrite(f"aligned_heads/if/face_{index}_0.jpg", aimg)


if __name__ == '__main__':
    Fire(align)
