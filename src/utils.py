import numpy as np


def rebin(image, k, axis=None, update_header=None):
    nx, ny = image.shape
    if axis == 0:
        if update_header is not None:
            update_header['NAXIS2'] = update_header['NAXIS2'] // k
            update_header['CRPIX2'] = (update_header['CRPIX2'] - 0.5) / k + 0.5
            update_header['CDELT2'] = update_header['CDELT2'] * k
        return np.mean(np.reshape(image[:nx // k * k, :], (nx // k, -1, ny)), axis=-2)
    elif axis == 1:
        if update_header is not None:
            update_header['NAXIS1'] = update_header['NAXIS1'] // k
            update_header['CRPIX1'] = (update_header['CRPIX1'] - 0.5) / k + 0.5
            update_header['CDELT1'] = update_header['CDELT1'] * k
        return np.mean(np.reshape(image[:, :ny // k * k], (nx, ny // k, -1)), axis=-1)
    else:
        return rebin(rebin(image, k, axis=0, update_header=update_header), k, axis=1, update_header=update_header)


def crop_grid(xi, yi, header):
    nx, ny = header['NAXIS2'], header['NAXIS1']
    x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1
    return xi[x0:x0 + nx, y0:y0 + ny] - x0, yi[x0:x0 + nx, y0:y0 + ny] - y0


def undistort(image, header, xd, yd, **kwargs):
    from interpolation import interp2d
    xd_, yd_ = crop_grid(xd, yd, header)
    return interp2d(image, xd_, yd_, **kwargs)

