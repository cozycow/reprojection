import numpy as np
from astropy.io import fits
from view import View, Map


def reproject(data, header, header_new=None, kind='synoptic',
              correct_mu=True, **kwargs):

    view = View.from_header(header)

    if header_new is not None:
        view_new = View.from_header(header_new)

    elif kind == 'synoptic':
        nx, ny = 720, 1800
        view_new = Map(nx, ny, projection='sine_lat')

    elif kind == 'north':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        Rsun = 0.45 * nx
        view_new = View(nx, ny, xc, yc, Rsun, -90, 90, 0)

    elif kind == 'south':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        Rsun = 0.45 * nx
        view_new = View(nx, ny, xc, yc, Rsun, 90, -90, 0)

    elif kind == 'warp':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        Rsun = 0.45 * nx
        view_new = View(nx, ny, xc, yc, Rsun, 0, 0, 0, rsun_arc=-45 * 3600)

    else:
        view_new = view.copy()

    view_new.update(inplace=True, **kwargs)

    data_, alpha = view_new.reproject(data, view, **kwargs)

    if correct_mu:
        data_ *= alpha

    return data_, alpha, view_new



def make_map(files, correct_crota=True, binning=1, **kwargs):
    '''
    Creates an averaged map by applying 'remap' routine to every dataset in files
    '''

    from utils import rebin

    coverage = 0
    mean_data = 0
    mean_mu = 0
    var_mu = 0
    var_data = 0
    cov_data_mu = 0

    map = None

    for file in files:
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
        data, alpha, view = reproject(data, header, return_alpha=True, correct_mu=False, **kwargs)

        if map is None:
            map = view.copy()

        mu = 1 / alpha

        weight = (~np.isnan(data)).astype(float) #* mu
        coverage += np.nan_to_num(weight)

        mean_mu += np.nan_to_num((mu - mean_mu) * weight / coverage)
        mean_data += np.nan_to_num((data - mean_data) * weight / coverage)

        var_mu += np.nan_to_num((mu ** 2 - var_mu) * weight / coverage)
        var_data += np.nan_to_num((data ** 2 - var_data) * weight / coverage)
        cov_data_mu += np.nan_to_num((data * mu - cov_data_mu) * weight / coverage)

    cov_data_mu[coverage == 0] = np.nan
    var_mu[coverage == 0] = np.nan
    var_data[coverage == 0] = np.nan
    mean_mu[coverage == 0] = np.nan
    mean_data[coverage == 0] = np.nan
    coverage[coverage == 0] = np.nan

    beta = cov_data_mu / var_mu
    variance = ((var_data - mean_data ** 2) - beta ** 2 * (var_mu - mean_mu ** 2)) / var_mu / (coverage - 1)

    return beta, variance, map

