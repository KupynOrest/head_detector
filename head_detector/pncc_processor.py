from typing import List, Optional

import head_detector.Sim3DR as Sim3DR
import numpy as np

from head_detector.utils import get_relative_path
from head_detector.head_info import HeadMetadata


def pncc(
    img: np.ndarray, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, with_bg_flag: bool = True
) -> np.ndarray:
    """
    Render a colored 3D face mesh

    Args:
        img: Image where to render 3D face, RGB image of [H,W,3] size
        vertices: List of 3D vertices [N,3]
        faces: List of faces [N,3]
        colors: List of RGB colors for each vertex, [N,3]
        with_bg_flag: If True, paint on top of the image, otherwise - on black background.

    Returns:
        Image of shape [H,W,3] shape
    """

    def _to_ctype(arr: np.ndarray) -> np.ndarray:
        if not arr.flags.c_contiguous:
            return arr.copy(order="C")
        return arr

    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)
    overlap = Sim3DR.rasterize(_to_ctype(vertices), _to_ctype(faces), _to_ctype(colors), bg=overlap)
    return overlap


def compute_ncc_color_codes(template_face: np.ndarray, subset_indexes: Optional[np.ndarray] = None) -> np.ndarray:
    if not isinstance(template_face, np.ndarray):
        raise ValueError(f"Argument template_face must be a numpy array, got type {type(template_face)}")
    if len(template_face.shape) != 2 or template_face.shape[1] != 3:
        raise ValueError(f"Argument template_face must have shape [N,3], got shape {template_face.shape}")
    if subset_indexes is not None and not isinstance(subset_indexes, np.ndarray):
        raise ValueError(f"Argument subset_indexes must be a numpy array, got type {type(subset_indexes)}")

    sub_vertices = template_face[subset_indexes] if subset_indexes is not None else template_face
    u_min = sub_vertices.min(axis=0, keepdims=True, initial=0)
    u_max = sub_vertices.max(axis=0, keepdims=True, initial=0)

    def normalize_to_unit(u: np.ndarray, min: np.ndarray, max: np.ndarray) -> np.ndarray:
        return (u - min) / (max - min)

    return normalize_to_unit(template_face, u_min, u_max)


class PNCCProcessor:
    def __init__(self):
        self.indices = np.load(get_relative_path('assets/flame_indices/head_w_ears.npy', __file__))
        self.triangles = np.load(get_relative_path('assets/full_faces.npy', __file__))
        self.triangles = np.array([x for x in self.triangles if all(vertex_index in self.indices for vertex_index in x)]).astype(np.int32)
        v_template = np.load(get_relative_path('assets/v_template.npy', __file__))
        self.colors = compute_ncc_color_codes(v_template, self.indices)

    def __call__(self, image: np.ndarray, heads: List[HeadMetadata]) -> np.ndarray:
        pncc_image = np.zeros_like(image)
        for head in heads:
            vertices = head.vertices_3d
            vertices[:, 2] *= -1
            current_pncc = pncc(pncc_image, vertices, self.triangles, self.colors)
            pncc_image[current_pncc.sum(2) != 0] = current_pncc[current_pncc.sum(2) != 0]
        return pncc_image
