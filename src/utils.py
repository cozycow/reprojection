import numpy as np


def __rebin_axis(y, q, axis=0):
    x_ = np.arange(0, y.shape[axis], q)

    new_shape = list(y.shape)
    new_shape[axis] = len(x_)
    y_ = np.zeros(new_shape)

    for i in range(0, len(x_) - 1):
        a = x_[i]
        b = x_[i + 1]
        a_ = np.ceil(a).astype(int)
        b_ = np.floor(b).astype(int)
        da = a_ - a
        db = b - b_
        np.put_along_axis(y_, np.expand_dims(i, tuple(range(0, len(y.shape)))),
                          np.expand_dims(
                              (np.nansum(np.take(y, np.arange(a_, b_), axis=axis), axis=axis) +
                               da * np.take(y, a_ - 1, axis=axis) +
                               db * np.take(y, b_, axis=axis)) / q,
                              axis=axis),
                          axis=axis)

    a = x_[-1]
    a_ = np.ceil(a).astype(int)
    da = a_ - a
    np.put_along_axis(y_, np.expand_dims(-1, tuple(range(0, len(y.shape)))),
                      np.expand_dims(
                          (np.nansum(np.take(y, np.arange(a_, y.shape[axis]), axis=axis), axis=axis) +
                           da * np.take(y, a_ - 1, axis=axis)) / (y.shape[axis] - a),
                          axis=axis),
                      axis=axis)
    return y_


def rebin(image, k, axis=None, update_header=None):
    def __update_header(header, k, axis=None):
        if axis is None:
            __update_header(header, k, axis=0)
            __update_header(header, k, axis=1)
        else:
            header[f'NAXIS{2-axis:d}'] = np.ceil(header[f'NAXIS{2-axis:d}'] / k)
            header[f'CRPIX{2-axis:d}'] = (header[f'CRPIX{2-axis:d}'] - 0.5) / k + 0.5
            header[f'CDELT{2-axis:d}'] = header[f'CDELT{2-axis:d}'] * k

    if update_header is not None:
        __update_header(update_header, k, axis=axis)

    if len(image.shape) == 2:
        if axis is not None:
            return __rebin_axis(image, float(k), axis=axis)
        else:
            return __rebin_axis(__rebin_axis(image, float(k), axis=0), float(k), axis=1)
    else:
        out = []
        for i in range(len(image)):
            out.append(rebin(image[i], k, axis=axis-1))
        return np.array(out)


def crop(image, header=None, x1=None, x2=None, y1=None, y2=None, **kwargs):
    if header is not None:
        x1, x2, y1, y2 = header['PXBEG2'] - 1, header['PXEND2'], header['PXBEG1'] - 1, header['PXEND1']
    nx, ny = x2 - x1 + 1, y2 - y1 + 1

    if (isinstance(image, np.ndarray) and (len(image.shape) > 1) and (image.shape[-2:] != (nx, ny)) and
            x1 is not None and x2 is not None and y1 is not None and y2 is not None):
        return image[..., x1:x2, y1:y2]
    else:
        return image
