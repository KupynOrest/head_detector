import os
import json
from dataclasses import dataclass
from typing import List, Optional

from annoy import AnnoyIndex
from deepface import DeepFace

from build_face_index import EMBEDDING_SIZE

DISTANCE_THRESHOLD = 0.23


@dataclass
class FaceVerificationResult:
    verified: bool
    distance: float
    image_path: str


class FaceRetrieval:
    def __init__(self, metadata_path: str, search_k: int = -1):
        self.search_k = search_k
        self.annoy_index = AnnoyIndex(EMBEDDING_SIZE, 'angular')
        self.annoy_index.load(os.path.join(metadata_path, "face_index.ann"))
        with open(os.path.join(metadata_path, "path_mapping.json"), "r") as file:
            self.path_mapping = json.load(file)

    def retrieve(self, image_path: str, num_results: int = 5) -> List[Optional[FaceVerificationResult]]:
        result = []
        embedding_objs = DeepFace.represent(img_path=image_path, model_name="ArcFace", enforce_detection=False)
        for embedding_obj in embedding_objs:
            embedding = embedding_obj["embedding"]
            if embedding_obj["face_confidence"] < 0.8:
                result.append(None)
                continue
            embedding_result = []
            matches, distances = self.annoy_index.get_nns_by_vector(embedding, num_results, search_k=self.search_k, include_distances=True)
            for match, distance in zip(matches, distances):
                embedding_result.append(FaceVerificationResult(
                    verified=distance < DISTANCE_THRESHOLD,
                    distance=distance,
                    image_path=self.path_mapping.get(str(match), str(match)))
                )
            result.append(embedding_result)
        return result
