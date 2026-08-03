import numpy as np
from astropy.io import fits
from view import View
from interpolation import interp2d


def reproject(data, header, header_new=None, **kwargs):
    '''
    Reprojects the data obtained from a view defined by 'header' to a view defined by 'header_new'.
    '''

    if header_new is None:
        header_ = header
    else:
        header_ = header_new

    view = View.from_header(header)
    view_new = View.from_header(header_).update(**kwargs)
    return view_new.reproject(data, view, **kwargs)


def remap(data, header, dlon=0.2, dlat=0.2, sine_lat=False, return_alpha=False, mu_thr=0.1, **kwargs):
    '''
    Remaps the data obtained from a view defined by 'header' to spherical (Carrington) coordinates.
    '''

    from transforms import ToSpherical

    view = View.from_header(header).update(**kwargs)
    transform = ~ToSpherical() - view.to_carrington(correct_mu=True, mu_thr=mu_thr, **kwargs)

    if sine_lat:
        grid = np.mgrid[-1:1 + dlat / 2:dlat, :360:dlon]
        grid[0] = np.arcsin(grid[0].clip(-1,1)) * 180 / np.pi
    else:
        grid = np.mgrid[-90:90 + dlat / 2:dlat, :360:dlon]

    grid, alpha = transform(grid)
    data_ = interp2d(data, *grid, **kwargs) * alpha
    if return_alpha:
        return data_, alpha
    else:
        return data_


def polar_view(data, header, pole='north', return_alpha=False, qr=0.45, mu_thr=0.1, **kwargs):
    nx, ny = 1024, 1024
    xc, yc = (nx - 1) / 2, (ny - 1) / 2
    Rsun = qr * nx

    if pole == 'north':
        view_new = View(nx, ny, xc, yc, Rsun, -90, 90, 0) ### North pole view
    else:
        view_new = View(nx, ny, xc, yc, Rsun, 90, -90, 0) ### South pole view

    grid = view_new.grid()
    view = View.from_header(header)

    transform = (view_new.to_carrington(correct_mu=True) -
                 view.to_carrington(correct_mu=True, mu_thr=mu_thr, **kwargs))

    grid, alpha = transform(grid)
    data_ = interp2d(data, *grid, **kwargs) * alpha
    if return_alpha:
        return data_, alpha
    else:
        return data_


def make_map(files, pow=4, polar=False, binning=1, **kwargs):
    '''
    Creates an averaged map by applying 'remap' routine to every dataset in files
    '''
    from utils import rebin

    mean, coverage = 0, 0
    for file in files:
        with fits.open(file) as hdul:
            header = hdul[0].header.copy()
            data = hdul[0].data.copy()

        data = rebin(data, binning, update_header=header)

        if polar:
            data, alpha = polar_view(data, header, return_alpha=True, **kwargs)
        else:
            data, alpha = remap(data, header, return_alpha=True, **kwargs)
        weight = (~np.isnan(data)) * alpha ** (-pow)

        coverage += np.nan_to_num(weight)
        mean += np.nan_to_num((data - mean) * weight / coverage)

    mean[coverage == 0] = np.nan
    coverage[coverage == 0] = np.nan
    return mean, coverage
