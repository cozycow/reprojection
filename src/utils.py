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


def crop(image, header=None, x1=None, x2=None, y1=None, y2=None, **kwargs):
    if header is not None:
        x1, x2, y1, y2 = header['PXBEG2'] - 1, header['PXEND2'], header['PXBEG1'] - 1, header['PXEND1']
    nx, ny = x2 - x1 + 1, y2 - y1 + 1

    if (isinstance(image, np.ndarray) and (len(image.shape) > 1) and (image.shape[-2:] != (nx, ny)) and
            x1 is not None and x2 is not None and y1 is not None and y2 is not None):
        return image[..., x1:x2, y1:y2]
    else:
        return image
