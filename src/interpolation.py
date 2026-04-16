import numpy as np


def interp2d(image, x, y, kind='bilinear', **kwargs):
    def __bilinear(z):
        return 1 - np.abs(z)

    def __bicubic(z):
        z_ = np.abs(z)
        return np.where(z_ <= 1, 1.5 * z_ ** 3 - 2.5 * z_ ** 2 + 1,
                        -0.5 * z_ ** 3 + 2.5 * z_ ** 2 - 4 * z_ + 2)

    if kind == 'bicubic':
        kernel = __bicubic
        nodes = range(-1,3)
    else:
        kernel = __bilinear
        nodes = range(0,2)

    nx, ny = image.shape
    x_ = np.nan_to_num(np.floor(x), nan=nx).astype(np.int16)
    y_ = np.nan_to_num(np.floor(y), nan=ny).astype(np.int16)
    dx, dy = x - x_, y - y_

    image_ = np.zeros_like(x).astype(np.float32)
    for i in nodes:
        kernel_ = kernel(i - dx)
        for j in nodes:
            xi, yj = x_ + i, y_ + j
            temp = image[xi % nx, yj % ny] * kernel_ * kernel(j - dy)
            image_ += temp

    return image_


def interpolate(f, x, x_new):
    idx = np.searchsorted(x, x_new).clip(1, len(x) - 1)

    xa = x[idx - 1]
    xb = x[idx]
    dx = xb - xa

    a, b = (xb - x_new) / dx, (x_new - xa) / dx

    fa = np.take_along_axis(f, idx - 1, axis=0)
    fb = np.take_along_axis(f, idx, axis=0)

    f_new = fa * a + fb * b
    return f_new
