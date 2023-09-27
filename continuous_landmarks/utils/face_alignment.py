import cv2
import numpy as np
import torch
from numpy.linalg import norm
from PIL import Image
from sklearn.preprocessing import normalize


def align_face(
    img, landmarks,
    eye_0, eye_1, mouth_0, mouth_1
):
    r90 = np.array([
        [0, 1],
        [-1, 0]
    ])

    xp = eye_1 - eye_0
    yp = 1/2*(eye_0 + eye_1) - 1/2*(mouth_0 + mouth_1)
    c = 1/2*(eye_0 + eye_1) - 0.1*yp
    s = max(4.0*norm(xp), 3.6*norm(yp))
    x = xp - r90 @ yp.T
    x /= norm(x)
    y = - r90 @ x.T

    theta = np.arccos(np.dot(x, [1, 0])) * 180 / np.pi

    if x[1] > 0:
        theta *= -1

    top_left = c - x * s/2 - y * s/2

    shift = np.hstack([np.eye(2), -top_left[:, None]])
    rot = cv2.getRotationMatrix2D(np.zeros(2), -theta, 1)
    M = rot @ np.vstack((shift, [0, 0, 1]))

    aligned_img = cv2.warpAffine(img, M, (int(s), int(s)))
    aligned_lms = np.hstack([landmarks, np.ones(len(landmarks))[:, None]]) @ M.T

    return aligned_img, aligned_lms