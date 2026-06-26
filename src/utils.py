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


def crop_grid(xi, yi, header):
    nx, ny = header['NAXIS2'], header['NAXIS1']
    x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1
    return xi[x0:x0 + nx, y0:y0 + ny] - x0, yi[x0:x0 + nx, y0:y0 + ny] - y0


def undistort(image, header, xd, yd, **kwargs):
    from interpolation import interp2d
    xd_, yd_ = crop_grid(xd, yd, header)
    return interp2d(image, xd_, yd_, **kwargs)


def remove_straylight(data, alpha=2., beta=1.5, epsilon=0.25, size=511, niter=3):
    from scipy.signal import fftconvolve
    nx, ny = data.shape[-2:]
    xi, yi = np.mgrid[-size:size + 1,-size:size + 1]
    ri = np.sqrt(xi ** 2 + yi ** 2)

    q = (1 + (ri / alpha) ** 2) ** (-beta)
    q /= np.sum(q)

    data_ = data.copy().reshape((-1, nx, ny))
    result = data_.copy()

    for _ in range(niter):
        for i in range(len(data_)):
            temp = fftconvolve(result[i], q, mode='same')
            result[i] = data_[i] - epsilon * (temp - data_[i])

    return result.reshape(data.shape)


def reflection_point_predict(header):
    px = [1.63114715e-06, 6.72511045e-03, 9.60448053e+02]
    py = [4.61830880e-06, -6.85005911e-03, 9.77508840e+02]

    r_sun = header['RSUN_ARC']
    dx, dy = header['PXBEG2'] - 1, header['PXBEG1'] - 1

    xr = np.polyval(px, r_sun) - dx
    yr = np.polyval(py, r_sun) - dy
    return xr, yr


def roll(image, dx, dy):
    nx, ny = image.shape
    image_ = np.zeros_like(image)
    x, y = int(round(dx)), int(round(dy))
    image_[max(x, 0): min(nx + x, nx), max(y, 0): min(ny + y, ny)] = image[max(-x, 0): min(nx - x, nx),
                                                                     max(-y, 0): min(ny - y, ny)]
    return image_


def reflect(image, xr, yr):
    nx, ny = image.shape
    return roll(image[::-1, ::-1], 2 * int(round(xr)) - nx + 1, 2 * int(round(yr)) - ny + 1)
