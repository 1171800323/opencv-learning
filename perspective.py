import math

import cv2
import numpy as np


def center_warp_perspective(img, H, center, size):

    P = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]],
                 dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))

    img = cv2.warpPerspective(img, M, size,
                              cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return img


def center_points_perspective(points, H, center):

    P = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]],
                 dtype=np.float32)
    M = P.dot(H).dot(np.linalg.inv(P))
    return M.dot(points)


def get_perspective_transform(rotate_angle, zoom, shear_angle, perspect):
    rotate_angle = rotate_angle * math.pi / 180.
    shear_x_angle = shear_angle[0] * math.pi / 180.
    shear_y_angle = shear_angle[1] * math.pi / 180.
    scale_w, scale_h = zoom
    perspect_x, perspect_y = perspect

    # 放缩
    H_scale = np.array([[scale_w, 0, 0], [0, scale_h, 0], [0, 0, 1]],
                       dtype=np.float32)
    # 旋转
    H_rotate = np.array([[math.cos(rotate_angle),
                          math.sin(rotate_angle), 0],
                         [-math.sin(rotate_angle),
                          math.cos(rotate_angle), 0], [0, 0, 1]],
                        dtype=np.float32)
    # 剪切：将所有点沿某一指定方向成比例地平移
    H_shear = np.array([[1, math.tan(shear_x_angle), 0],
                        [math.tan(shear_y_angle), 1, 0], [0, 0, 1]],
                       dtype=np.float32)
    # 透视变换
    H_perspect = np.array([[1, 0, 0], [0, 1, 0], [perspect_x, perspect_y, 1]],
                          dtype=np.float32)

    H = H_rotate.dot(H_shear).dot(H_scale).dot(H_perspect)

    return H


if __name__ == "__main__":
    img_h, img_w = 60, 200
    img = 127 * np.ones((img_h, img_w), dtype=np.uint8)
    H = get_perspective_transform(rotate_angle=15,
                                  zoom=[0.3, 0.5],
                                  shear_angle=[-0.49, -3.8],
                                  perspect=[0.0009, -0.0009])
    # H = get_perspective_transform(rotate_angle=-89,
    #                               zoom=[0.5, 0.5],
    #                               shear_angle=[0, 0],
    #                               perspect=[0, 0])
    print(H)
    img_center = (img_w / 2, img_h / 2)
    points = np.ones((3, 4), dtype=np.float32)
    points[:2, 0] = np.array([0, 0], dtype=np.float32).T
    points[:2, 1] = np.array([img_w, 0], dtype=np.float32).T
    points[:2, 2] = np.array([img_w, img_h], dtype=np.float32).T
    points[:2, 3] = np.array([0, img_h], dtype=np.float32).T

    perspected_points = center_points_perspective(points, H, img_center)
    print(perspected_points)

    perspected_points[0, :] /= perspected_points[2, :]
    perspected_points[1, :] /= perspected_points[2, :]

    print(perspected_points)

    canvas_w = int(
        2 * max(img_center[0], img_center[0] - np.min(perspected_points[0, :]),
                np.max(perspected_points[0, :]) - img_center[0])) + 10
    canvas_h = int(
        2 * max(img_center[1], img_center[1] - np.min(perspected_points[1, :]),
                np.max(perspected_points[1, :]) - img_center[1])) + 10

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    tly = (canvas_h - img_h) // 2
    tlx = (canvas_w - img_w) // 2
    canvas[tly:tly + img_h, tlx:tlx + img_w] = img
    cv2.imshow("canvas0", canvas)

    canvas_center = (canvas_w // 2, canvas_h // 2)
    canvas_size = (canvas_w, canvas_h)
    print(canvas_size)
    canvas = center_warp_perspective(canvas, H, canvas_center, canvas_size)
    cv2.imshow("canvas1", canvas)

    perspected_points[0] += tlx
    perspected_points[1] += tly
    perspected_points = np.int64(perspected_points)

    print(perspected_points)

    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for i in range(4):
        cv2.circle(canvas, (perspected_points[0, i], perspected_points[1, i]),
                   radius=1,
                   color=(0, 0, 255),
                   thickness=2)

    cv2.imshow("img", img)
    cv2.imshow("canvas2", canvas)
    cv2.waitKey()
