import numpy as np
from typing import Union, Tuple
import cv2
import torch
from yolo_head.flame import FlameParams, rot_mat_from_6dof, RPY

NORMALIZED_IMAGE_SIZE = 1024
IMG_SIZE = 256
DISTANCE_OUTER_EYES_CORNERS_NORMALIZED = 116
LEYE_INDEX_3D = 2437
REYE_INDEX_3D = 1175
SKULL_CENTER_LEFT_END_IDX = 567
SKULL_CENTER_RIGHT_END_IDX = 1962


def homogeneous_matrix_2D(mat: np.ndarray) -> np.ndarray:
    if mat.shape == (2, 2):
        mat = np.concatenate((mat, np.array([[0.0], [0.0]])), -1)
    transform = np.concatenate((mat, np.array([[0.0, 0.0, 1.0]])))
    return transform


def homogeneous_transform_2D(p: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if p.shape == (2,):
        p = np.expand_dims(p, 1)
    p_homo = np.concatenate((p, np.array([[1.0]])))
    return np.dot(transform, p_homo)[:2]


def get_scale_matrix(sx: float, sy: Union[float, None] = None) -> np.ndarray:
    if not sy:
        sy = sx
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]])


def get_rotation_matrix(angle: float, center: Tuple[int, int] = (0, 0)) -> np.ndarray:
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def get_translation_matrix(tx: Union[float, int], ty: Union[float, int, None] = None) -> np.ndarray:
    if not ty:
        ty = tx
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])


def get_scale(v: np.ndarray) -> np.ndarray:
    return DISTANCE_OUTER_EYES_CORNERS_NORMALIZED / np.linalg.norm(v[LEYE_INDEX_3D] - v[REYE_INDEX_3D])


def get_skull_center(v: np.ndarray) -> np.ndarray:
    return (v[SKULL_CENTER_LEFT_END_IDX] + v[SKULL_CENTER_RIGHT_END_IDX]) / 2


def get_zero_to_normalized_transform(zero_vertices: np.ndarray) -> np.ndarray:
    zero_projected_vertices = zero_vertices[..., :2]
    # head center has a small translation on y to match previous placement with 2d keypoints
    base_mesh_center_x, base_mesh_center_y = 0, -0.01
    base_mesh_scale = get_scale(zero_projected_vertices)

    base_mesh_transform = cv2.getRotationMatrix2D((base_mesh_center_x, base_mesh_center_y), 0, base_mesh_scale)
    base_mesh_transform[0, 2] += NORMALIZED_IMAGE_SIZE / 2
    base_mesh_transform[1, 2] += NORMALIZED_IMAGE_SIZE / 2
    return base_mesh_transform


def get_zero_to_user_transform(
    flame_params: FlameParams, user_vertices: np.ndarray, zero_vertices: np.ndarray, rpy: RPY
) -> np.ndarray:
    user_mesh_center_x, user_mesh_center_y = ((flame_params.translation + 1) / 2 * 256)[0, :2]
    if np.abs(rpy.yaw) <= 45:
        user_mesh_center_y = (1.0 - 0.5 * np.sin(np.radians(rpy.pitch))) * user_mesh_center_y
    scale = np.linalg.norm(user_vertices[LEYE_INDEX_3D] - user_vertices[REYE_INDEX_3D]) / np.linalg.norm(
        zero_vertices[LEYE_INDEX_3D] - zero_vertices[REYE_INDEX_3D]
    )

    scale_matrix = get_scale_matrix(scale)
    roll = rpy.roll
    rotation_matrix = get_rotation_matrix(roll)
    translation_matrix = get_translation_matrix(user_mesh_center_x, user_mesh_center_y)
    transform = (
        homogeneous_matrix_2D(translation_matrix)
        @ homogeneous_matrix_2D(rotation_matrix)
        @ homogeneous_matrix_2D(scale_matrix)
    )
    transform = transform[:2]
    return transform


def flame_params_skull_center(flame_params: FlameParams, img_size: int) -> Tuple[int, int]:
    scull_center = (flame_params.translation + 1.0) / 2.0 * img_size
    scull_center = scull_center[0].numpy()
    return int(scull_center[0]), int(scull_center[1])


def euler_angles_to_rotation_matrix(
    roll: Union[float, int], pitch: Union[float, int], yaw: Union[float, int]
) -> np.ndarray:
    theta = (pitch, yaw, roll)
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = R_z @ R_y @ R_x
    return R


def get_rotation_mat(
    img: np.ndarray, img_center: Tuple[int, int], angle: Union[float, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = img.shape[:2]
    rotation_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - img_center[0]
    rotation_mat[1, 2] += bound_h / 2 - img_center[1]
    return rotation_mat, (bound_w, bound_h)


def rotate_3dmm_translation(
    flame_params: FlameParams, scull_center: Tuple[int, int], rotation_mat: np.ndarray, img_size: int
) -> None:
    translate_x, translate_y = scull_center
    new_x, new_y = rotation_mat @ [translate_x, translate_y, 1]
    new_x = new_x / img_size * 2 - 1
    new_y = new_y / img_size * 2 - 1
    flame_params.translation[0, 0] = new_x
    flame_params.translation[0, 1] = new_y


def rotate_3dmm_rotation(flame_params: FlameParams, rpy: RPY) -> None:
    orig_mat = euler_angles_to_rotation_matrix(rpy.roll * np.pi / 180, 0, 0)
    rotate_3dmm_rotation_6dof(flame_params, orig_mat)


def rotate_3dmm_rotation_6dof(flame_params: FlameParams, orig_mat: np.ndarray) -> None:
    roll_mat = rot_mat_from_6dof(flame_params.rotation)[0]
    mat = torch.from_numpy(orig_mat).float() @ roll_mat
    flame_params.rotation[0, :] = torch.cat((mat.T[[0]], mat.T[[1]]), -1)


def rotate_3dmm(
    flame_params: FlameParams, scull_center: Tuple[int, int], rotation_mat: np.ndarray, rpy: RPY, img_size: int
) -> FlameParams:
    rotate_3dmm_translation(flame_params, scull_center, rotation_mat, img_size)
    rotate_3dmm_rotation(flame_params, rpy)
    return flame_params


def vertically_align(
    img: np.ndarray, flame_params: FlameParams, rpy: RPY, img_size: int
) -> Tuple[np.ndarray, FlameParams]:
    scull_center = flame_params_skull_center(flame_params, img_size)
    rot_mat, bounds = get_rotation_mat(img, scull_center, -rpy.roll)

    vertical_img = cv2.warpAffine(img, rot_mat, bounds, flags=cv2.INTER_LINEAR)
    vertical_params = rotate_3dmm(flame_params, scull_center, rot_mat, rpy, img_size)
    return vertical_img, vertical_params
