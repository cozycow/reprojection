import numpy as np
from astropy.io import fits
from view import View


def reproject(data, header, header_new=None, correct_mu=True, **kwargs):
    '''
    Reprojects the data obtained from a view defined by 'header' to a view defined by 'header_new'.
    '''

    if header_new is None:
        header_ = header
    else:
        header_ = header_new

    view = View.from_header(header)
    view_new = View.from_header(header_).update(**kwargs)

    data_, alpha = view_new.reproject(data, view, **kwargs)
    if correct_mu:
        data_ *= alpha

    return data_


def remap(data, header, nx=720, ny=1800,
          return_alpha=False, correct_mu=True, **kwargs):
    '''
    Remaps the data obtained from a view defined by 'header' to spherical (Carrington) coordinates.
    '''

    view = View.from_header(header).update(**kwargs)
    data_, alpha = view.remap(data, nx=nx, ny=ny, **kwargs)

    if correct_mu:
        data_ *= alpha

    if return_alpha:
        return data_, alpha
    else:
        return data_


def polar_view(data, header, pole='north', return_alpha=False,
               qr=0.45, correct_mu=True, **kwargs):
    nx, ny = 1024, 1024
    xc, yc = (nx - 1) / 2, (ny - 1) / 2
    Rsun = qr * nx

    if pole == 'north':
        view_new = View(nx, ny, xc, yc, Rsun, -90, 90, 0) ### North pole view
    else:
        view_new = View(nx, ny, xc, yc, Rsun, 90, -90, 0) ### South pole view

    view = View.from_header(header)
    data_, alpha = view_new.reproject(data, view, **kwargs)

    if correct_mu:
        data_ *= alpha

    if return_alpha:
        return data_, alpha
    else:
        return data_


def make_map(files, polar=False, correct_bias=False, correct_crota=True, binning=1, **kwargs):
    '''
    Creates an averaged map by applying 'remap' routine to every dataset in files
    '''

    from utils import rebin

    coverage = 0
    mean_data = 0
    mean_mu = 0
    var_mu = 0
    cov_data_mu = 0

    for file in files:
        mean_mu_ = mean_mu + 0.

        with fits.open(file) as hdul:
            if 'hmi' in file:
                data = hdul[1].data.copy()
                header = hdul[1].header.copy()
            else: # 'fdt'
                data = hdul[0].data.copy()
                header = hdul[0].header.copy()

                if correct_crota:
                    header['CROTA'] = header['CROTA'] + 0.25 ### crota correction

        data = rebin(data, binning, update_header=header)

        if polar:
            data, alpha = polar_view(data, header, return_alpha=True, correct_mu=False, **kwargs)
        else:
            data, alpha = remap(data, header, return_alpha=True, correct_mu=False, **kwargs)

        mu = 1 / alpha

        weight = (~np.isnan(data)).astype(float) #* mu
        coverage += np.nan_to_num(weight)

        mean_mu += np.nan_to_num((mu - mean_mu) * weight / coverage)
        mean_data += np.nan_to_num((data - mean_data) * weight / coverage)

        if correct_bias:
            var_mu += np.nan_to_num((mu - mean_mu) * (mu - mean_mu_) * weight)
            cov_data_mu += np.nan_to_num((data - mean_data) * (mu - mean_mu_) * weight)
        else:
            var_mu += np.nan_to_num((mu ** 2 - var_mu) * weight / coverage)
            cov_data_mu += np.nan_to_num((data * mu - cov_data_mu) * weight / coverage)

    cov_data_mu[coverage == 0] = np.nan
    var_mu[coverage == 0] = np.nan
    coverage[coverage == 0] = np.nan

    Q = cov_data_mu / var_mu

    return Q, coverage

