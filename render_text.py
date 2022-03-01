import cv2
import numpy as np

from utils import (find_min_area_rect, paint_boundingbox, perspective,
                   pillow_make_standard_text)

img = cv2.imread("back.jpg")

boxes = [[[74, 255], [545, 383], [528, 457], [53, 339]],
         [[44, 145], [334, 58], [358, 143], [66, 226]]]
texts = ["退休", "工作"]

paint_img = paint_boundingbox(img=img, boxes=boxes)
cv2.imshow("paint_img", paint_img)
cv2.waitKey()

for box, text in zip(boxes, texts):
    new_box = np.array(box, dtype=np.int64)
    min_area_rect_box = find_min_area_rect(new_box)
    print(min_area_rect_box)
    back_img, M_inverse = perspective(img, min_area_rect_box)
    cv2.imshow("back", back_img)
    cv2.waitKey()

    fake_img, text_mask = pillow_make_standard_text("fonts/zh_bold.ttc",
                                                    text,
                                                    back_img.shape[:2],
                                                    fg_col=(209, 222, 223),
                                                    bg_img=back_img)
    cv2.imshow("fake_img", fake_img)
    cv2.waitKey()
    h, w = img.shape[:2]
    fake_inverse = cv2.warpPerspective(fake_img, M_inverse, (w, h))
    cv2.imshow("fake_inverse", fake_inverse)
    cv2.waitKey()
    mask = cv2.warpPerspective(text_mask, M_inverse, (w, h))
    cv2.imshow("mask", mask)
    cv2.waitKey()
    img[mask != 0] = fake_inverse[mask != 0]
    cv2.imshow("img", img)
    cv2.waitKey()
