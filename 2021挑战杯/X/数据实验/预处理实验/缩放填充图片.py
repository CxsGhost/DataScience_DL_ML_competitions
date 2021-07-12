import cv2
import numpy as np

img = cv2.imread('W0003_0027.png', cv2.IMREAD_GRAYSCALE)
size = img.shape
new_wid = 416
new_hig = size[0] * new_wid / size[1]
re_img = cv2.resize(img, (new_wid, int(new_hig)))


def pad_image(image, h, w, size):
    pad_image = image.copy()
    print(size)
    pad_h = max(h - size[0], 0) // 2
    pad_w = max(w - size[1], 0)
    print(pad_w, pad_h)
    if pad_h > 0 or pad_w > 0:
        # 选择用固定值padvalue填充，[top, bottom, left, right]
        pad_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w,
                                       cv2.BORDER_CONSTANT,
                                       value=50)
    return pad_image


pad_img = pad_image(re_img, 416, 416, re_img.shape)
cv2.imwrite('suo_xiao_pad.png', pad_img)
