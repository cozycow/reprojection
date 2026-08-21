import numpy as np


def rebin(image, k, axis=None, update_header=None):
    def __update_header(header, k, axis=None):
        if axis is None:
            __update_header(header, k, axis=0)
            __update_header(header, k, axis=1)
        else:
            header[f'NAXIS{2-axis:d}'] = header[f'NAXIS{2-axis:d}'] // k
            header[f'CRPIX{2-axis:d}'] = (header[f'CRPIX{2-axis:d}'] - 0.5) / k + 0.5
            header[f'CDELT{2-axis:d}'] = header[f'CDELT{2-axis:d}'] * k

    if update_header is not None:
        __update_header(update_header, k, axis=axis)

    if len(image.shape) == 2:
        nx, ny = image.shape
        if axis == 0:
            return np.mean(np.reshape(image[:nx // k * k, :], (nx // k, -1, ny)), axis=-2)
        elif axis == 1:
            return np.mean(np.reshape(image[:, :ny // k * k], (nx, ny // k, -1)), axis=-1)
        else:
            return rebin(rebin(image, k, axis=0), k, axis=1)
    else:
        out = []
        for i in range(len(image)):
            out.append(rebin(image[i], k, axis=axis))
        return np.array(out)
