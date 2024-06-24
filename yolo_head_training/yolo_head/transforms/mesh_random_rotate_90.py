import random
import numpy as np

from super_gradients.common.registry.registry import register_transform
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh
from super_gradients.training.transforms.keypoints.abstract_keypoints_transform import AbstractKeypointTransform

from ..mesh_sample import MeshEstimationSample


@register_transform(name="MeshRandomRotate90")
class MeshRandomRotate90(AbstractKeypointTransform):
    """
    Apply 90 degree rotations to the sample.
    """

    def __init__(
        self,
        prob: float = 0.5,
    ):
        """
        Initialize transform

        :param prob (float): Probability of applying the transform
        """
        super().__init__()
        self.prob = prob

    def apply_to_sample(self, sample: MeshEstimationSample) -> MeshEstimationSample:
        """
        :param   sample: Input PoseEstimationSample
        :return:         Result of applying the transform
        """

        if random.random() < self.prob:
            factor = random.randint(0, 3)

            image_rows, image_cols = sample.image.shape[:2]

            sample.image = self.apply_to_image(sample.image, factor)
            sample.rotation_matrix = self.apply_to_matrix(sample.rotation_matrix, factor)
            sample.vertices_2d = self.apply_to_keypoints(sample.vertices_2d, factor, image_rows, image_cols)

            if sample.bboxes_xywh is not None:
                sample.bboxes_xywh = self.apply_to_bboxes(sample.bboxes_xywh, factor, image_rows, image_cols)

        return sample

    @classmethod
    def apply_to_matrix(cls, rotation_matrix: np.ndarray, factor: int) -> np.ndarray:
        """
        Rotate 3x3 rotation matrix by 90 degrees

        :param rotation_matrix: 3x3 rotation matrix
        :param factor:          Number of 90 degree rotations to apply. Order or rotation matches np.rot90
        :return:               Rotated 3x3 rotation matrix
        """
        angle = np.radians(90 * factor)
        c = np.cos(angle)
        s = np.sin(angle)
        for index in range(rotation_matrix.shape[0]):
            rotation_matrix[index] = np.dot(rotation_matrix[index],  np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
        return rotation_matrix

    @classmethod
    def apply_to_image(cls, image: np.ndarray, factor: int) -> np.ndarray:
        """
        Rotate image by 90 degrees

        :param image:  Input image
        :param factor: Number of 90 degree rotations to apply. Order or rotation matches np.rot90
        :return:       Rotated image
        """
        return np.rot90(image, factor)

    @classmethod
    def apply_to_bboxes(cls, bboxes_xywh: np.ndarray, factor, rows: int, cols: int) -> np.ndarray:
        """

        :param bboxes: (N, 4) array of bboxes in XYWH format
        :param factor: Number of 90 degree rotations to apply. Order or rotation matches np.rot90
        :param rows:   Number of rows (image height) of the original (input) image
        :param cols:   Number of cols (image width) of the original (input) image
        :return:       Transformed bboxes in XYWH format
        """
        from super_gradients.training.transforms.transforms import DetectionRandomRotate90

        bboxes_xyxy = xywh_to_xyxy(bboxes_xywh, image_shape=None)
        bboxes_xyxy = DetectionRandomRotate90.xyxy_bbox_rot90(bboxes_xyxy, factor, rows, cols)
        return xyxy_to_xywh(bboxes_xyxy, image_shape=None)

    @classmethod
    def apply_to_keypoints(cls, keypoints: np.ndarray, factor, rows: int, cols: int) -> np.ndarray:
        """

        :param keypoints: Input keypoints array of [Num Instances, Num Joints, 3] shape.
                          Keypoints has format (x, y, visibility)
        :param factor:    Number of 90 degree rotations to apply. Order or rotation matches np.rot90
        :param rows:      Number of rows (image height) of the original (input) image
        :param cols:      Number of cols (image width) of the original (input) image
        :return:          Transformed keypoints array of [Num Instances, Num Joints, 3] shape.
        """
        x, y, v = keypoints[:, :, 0], keypoints[:, :, 1], keypoints[:, :, 2]

        if factor == 0:
            keypoints = x, y, v
        elif factor == 1:
            keypoints = y, cols - x - 1, v
        elif factor == 2:
            keypoints = cols - x - 1, rows - y - 1, v
        elif factor == 3:
            keypoints = rows - y - 1, x, v
        else:
            raise ValueError("Parameter n must be in set {0, 1, 2, 3}")
        return np.stack(keypoints, axis=-1)

    def __repr__(self):
        return self.__class__.__name__ + f"(prob={self.prob})"

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__} does not have equivalent preprocessing because it is non-deterministic.")
