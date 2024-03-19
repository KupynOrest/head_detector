import math

import super_gradients
import cv2
import albumentations as A
import numpy as np
from PIL import Image


def draw_bodypose(canvas: np.ndarray, keypoints) -> np.ndarray:
    stickwidth = 8

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5],
        [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16],
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    visible_keypoints = []
    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1[2] < 0.3 or keypoint2[2] < 0.3:
            continue

        Y = np.array([keypoint1[0], keypoint2[0]])
        X = np.array([keypoint1[1], keypoint2[1]])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])
        visible_keypoints.append((Y[0], X[0], k1_index))
        visible_keypoints.append((Y[1], X[1], k2_index))

    #create set from list of keypoints to draw only once
    visible_keypoints = set(visible_keypoints)
    for keypoint in visible_keypoints:
        x, y = keypoint[0], keypoint[1]
        x = int(x)
        y = int(y)
        cv2.circle(canvas, (int(x), int(y)), stickwidth, colors[keypoint[2] - 1], thickness=-1)
    return canvas


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


class PoseProcessor:
    def __init__(self, img_size: int = 1024):
        self.model = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
        self.transform = A.SmallestMaxSize(max_size=img_size, always_apply=True)

    @staticmethod
    def median_point(point_a, point_b):
        return (point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2, (point_a[2] + point_b[2]) / 2,

    def remap_keypoints(self, keypoints):
        remapped_keypoints = [
            keypoints[0],
            self.median_point(keypoints[5], keypoints[6]),
            keypoints[6],
            keypoints[8],
            keypoints[10],
            keypoints[5],
            keypoints[7],
            keypoints[9],
            keypoints[12],
            keypoints[14],
            keypoints[16],
            keypoints[11],
            keypoints[13],
            keypoints[15],
            keypoints[2],
            keypoints[1],
            keypoints[4],
            keypoints[3],
        ]
        return remapped_keypoints

    def process(self, image: np.ndarray) -> Image:
        image = self.transform(image=image)["image"]
        predictions = self.model.predict(image, conf=0.5)
        poses = predictions.prediction.poses
        canvas = np.zeros_like(image)
        for pose in poses:
            canvas = draw_bodypose(canvas, self.remap_keypoints(pose))
        return Image.fromarray(canvas[:, :, ::-1])

    def __call__(self, image: np.ndarray) -> Image:
        return self.process(image)
