import numpy as np

def bilinear(image, x, y):
    nx, ny = image.shape

    x_ = np.nan_to_num(np.floor(x), nan=nx).astype(np.int16)
    y_ = np.nan_to_num(np.floor(y), nan=ny).astype(np.int16)
    dx, dy = x - x_, y - y_

    image_ = np.zeros_like(x).astype(np.float32)
    for i in [0, 1]:
        for j in [0, 1]:
            q = np.abs((1 - i - dx) * (1 - j - dy))
            xi, yj = x_ + i, y_ + j
            temp = image[xi % nx, yj % ny] * q
            image_ += temp

    return image_
