import os
from typing import List

import numpy as np

from head_detector.draw_utils import draw_3d_landmarks, draw_2d_landmarks, draw_pose, draw_bboxes
from head_detector.head_info import HeadMetadata
from head_detector.pncc_processor import PNCCProcessor
from head_detector.utils import vertically_align, refined_head_bbox, extend_to_rect, extend_bbox ,get_relative_path


DRAW_MAPPING = {
    "landmarks": [draw_3d_landmarks],
    "points": [draw_2d_landmarks],
    "pose": [draw_pose],
    'full': [draw_bboxes, draw_3d_landmarks],
    'bbox': [draw_bboxes],
}
MAX_YAW = 60


class MeshSaver:
    def __init__(self) -> None:
        self.triangles = np.load(get_relative_path('assets/full_faces.npy', __file__)) + 1.

    def __call__(self, vertices: np.ndarray, output_path: str) -> None:
        """
        mesh: tuple (vertices, faces)
        Vertices: [N, 3]. Faces: [N, 3], 1st vertex has index '1', not '0'.
        """
        with open(output_path, 'w') as f:
            for vertex in vertices:
                f.write(f'v %.8f %.8f %.8f\n' % tuple(vertex))
            for face in self.triangles:
                f.write('f %d %d %d\n' % tuple(face))


class PredictionResult:
    def __init__(self, original_image: np.ndarray, heads: List[HeadMetadata]):
        self.original_image = original_image
        self.heads = heads
        self.pncc_processor = PNCCProcessor()
        self.mesh_saver = MeshSaver()

    def draw(self, method: str = 'full'):
        image = self.original_image.copy()
        draw_methods = DRAW_MAPPING[method]
        for head in self.heads:
            for draw_method in draw_methods:
                image = draw_method(image, head)
        return image

    def get_pncc(self):
        return self.pncc_processor(self.original_image, self.heads)

    def get_aligned_heads(self):
        head_images = []
        for head in self.heads:
            head_image = self.original_image.copy()
            pred_pose = head.head_pose
            roll = pred_pose.roll
            vertices = head.vertices_3d
            if np.abs(pred_pose.yaw) < MAX_YAW:
                head_image, vertices = vertically_align(head_image, vertices, head.flame_params, roll)
            head_bbox = refined_head_bbox(vertices)
            head_bbox = extend_to_rect(extend_bbox([head_bbox.x, head_bbox.y, head_bbox.w, head_bbox.h], offset=0.1))
            x, y, w, h = head_bbox
            crop = head_image[y:y + h, x:x + w]
            head_images.append(crop)
        return head_images

    def save_meshes(self, save_folder: str):
        os.makedirs(save_folder, exist_ok=True)
        for i, head in enumerate(self.heads):
            vertices = head.vertices_3d
            points = []
            points.extend(np.take(vertices, np.array(self.pncc_processor.indices), axis=0))
            self.mesh_saver(head.vertices_3d, os.path.join(save_folder, f"head_{i}.obj"))

    def __repr__(self):
        return f"PredictionResult(original_image={self.original_image.shape}, num heads={len(self.heads)})"
