import numpy as np


def rebin(image, k, axis=None, update_header=None):
    nx, ny = image.shape
    if axis == 0:
        return np.mean(np.reshape(image[:nx // k * k, :], (nx // k, -1, ny)), axis=-2)
    elif axis == 1:
        return np.mean(np.reshape(image[:, :ny // k * k], (nx, ny // k, -1)), axis=-1)
    else:
        if update_header is not None:
            update_header['NAXIS1'] = update_header['NAXIS1'] // k
            update_header['NAXIS2'] = update_header['NAXIS2'] // k
            update_header['CRPIX1'] = (update_header['CRPIX1'] - 1) / k + 1
            update_header['CRPIX2'] = (update_header['CRPIX2'] - 1) / k + 1
            update_header['CDELT1'] = update_header['CDELT1'] * k
            update_header['CDELT2'] = update_header['CDELT2'] * k
        return rebin(rebin(image, k, axis=0), k, axis=1)
