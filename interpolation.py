import numpy as np

def bilinear(image, x, y, nan=np.nan, periodic_x=False, periodic_y=False, **kwargs):
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

            if not periodic_x:
                temp[np.any([xi < 0,xi >= nx], axis=0)] = nan
            if not periodic_y:
                temp[np.any([yj < 0,yj >= ny], axis=0)] = nan
            image_ += temp

    return image_
