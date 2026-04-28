import numpy as np


def interp2d(image, x, y, kind='bicubic', roll=False, **kwargs):

    def __nearest(z):
        return np.where(np.abs(z) <= 0.5, 1, 0)

    def __bilinear(z):
        return 1 - np.abs(z)

    def __bicubic(z):
        z_ = np.abs(z)
        return np.where(z_ <= 1, 1.5 * z_ ** 3 - 2.5 * z_ ** 2 + 1,
                        -0.5 * z_ ** 3 + 2.5 * z_ ** 2 - 4 * z_ + 2)

    def __quadratic(z):
        z_ = np.abs(z)
        return np.where(z_ <= 1, 4 * z_ ** 3 / 3 - 7 * z_ ** 2 / 3 + 1,
                        np.where(z_ <= 2, -7 * z_ ** 3 / 12 + 3 * z_ ** 2 - 59 * z_ / 12 + 15 / 6,
                                 z_ ** 3 / 12 - 2 * z_ ** 2 / 3 + 21 * z_ / 12 - 3 / 2))

    if kind == 'quadratic':
        kernel = __quadratic
        nodes = range(-2,4)
    elif kind == 'bicubic':
        kernel = __bicubic
        nodes = range(-1,3)
    elif kind == 'bilinear':
        kernel = __bilinear
        nodes = range(0,2)
    else:
        kernel = __nearest
        nodes = range(0,2)

    nx, ny = image.shape
    x_ = np.nan_to_num(np.floor(x), nan=nx).astype(np.int16)
    y_ = np.nan_to_num(np.floor(y), nan=ny).astype(np.int16)
    dx, dy = x - x_, y - y_

    image_ = 0
    for i in nodes:
        kernel_ = kernel(i - dx)
        for j in nodes:
            if roll:
                temp = np.roll(image, (x_ + i, y_ + j), axis=(0, 1))
            else:
                temp = image[(x_ + i) % nx, (y_ + j) % ny]
            image_ += temp * kernel_ * kernel(j - dy)

    return image_


def interp1d(f, x, x_new):
    idx = np.searchsorted(x, x_new).clip(1, len(x) - 1)

    xa = x[idx - 1]
    xb = x[idx]
    dx = xb - xa

    a, b = (xb - x_new) / dx, (x_new - xa) / dx

    fa = np.take_along_axis(f, idx - 1, axis=0)
    fb = np.take_along_axis(f, idx, axis=0)

    f_new = fa * a + fb * b
    return f_new
