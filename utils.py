import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def paint_boundingbox(img, boxes, color=(0, 0, 255)):
    new_img = img.copy()
    for box in boxes:
        box = np.array([box], dtype=np.int64)
        cv2.drawContours(new_img,
                         box,
                         0,
                         color=color,
                         thickness=1,
                         lineType=cv2.LINE_AA)

    return new_img


def find_min_area_rect(points):
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    new_box = []
    for point in points:
        d0 = calculate_distance(point, box[0])
        d1 = calculate_distance(point, box[1])
        d2 = calculate_distance(point, box[2])
        d3 = calculate_distance(point, box[3])

        idx = np.argmin([d0, d1, d2, d3])
        new_box.append(box[idx])

    return new_box


def calculate_distance(p1, p2):
    d_x = p1[0] - p2[0]
    d_y = p1[1] - p2[1]
    return math.sqrt(d_x**2 + d_y**2)


def perspective(img, points):
    pts1 = np.float32(points)

    w = calculate_distance(pts1[0], pts1[1])
    h = calculate_distance(pts1[1], pts1[2])

    w = np.int64(w)
    h = np.int64(h)

    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (w, h))

    M_inverse = cv2.getPerspectiveTransform(pts2, pts1)
    return dst, M_inverse


def pillow_make_standard_text(font_path,
                              text,
                              shape,
                              padding=0.1,
                              init_fontsize=25,
                              fg_col=None,
                              bg_img=None):
    pre_remain = None
    fontsize = init_fontsize

    if padding < 1:
        border = int(min(shape) * padding)
    else:
        border = int(padding)
    target_shape = tuple(np.array(shape) - 2 * border)
    while True:
        ttf = ImageFont.truetype(font_path, size=fontsize)

        chars_w, chars_h = ttf.getsize(text)

        res_shape = tuple(np.array((chars_h, chars_w)))
        remain = np.min(np.array(target_shape) - np.array(res_shape))

        if pre_remain is not None:
            m = pre_remain * remain
            if m <= 0:
                if m < 0 and remain < 0:
                    fontsize -= 1
                if m == 0 and remain != 0:
                    if remain < 0:
                        fontsize -= 1
                    elif remain > 0:
                        fontsize += 1
                break
        if remain < 0:
            if fontsize == 2:
                break
            fontsize -= 1
        else:
            fontsize += 1
        pre_remain = remain

    ttf = ImageFont.truetype(font_path, size=fontsize)
    chars_w, chars_h = ttf.getsize(text)

    tly, tlx = int(0.5 * (shape[0] - chars_h) // 2), int(
        (shape[1] - chars_w) // 2)

    B, G, R = fg_col
    bg_img = Image.fromarray(cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB))
    ttf = ImageFont.truetype(font_path, size=fontsize)

    img_draw = ImageDraw.Draw(bg_img)

    img_draw.text((tlx, tly), text, font=ttf, fill=(R, G, B))

    text_mask = np.zeros(shape, dtype=np.uint8)
    text_mask[tly:tly + chars_h, tlx:tlx + chars_w] = 255

    return cv2.cvtColor(np.asarray(bg_img), cv2.COLOR_RGB2BGR), text_mask