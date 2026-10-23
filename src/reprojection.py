import numpy as np
from astropy.io import fits
from view import View
from scipy.ndimage import map_coordinates


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


def remap(data, header, nx=720, ny=1800, sine_lat=False,
          staggered_lat=False, return_alpha=False, correct_mu=True, **kwargs):
    '''
    Remaps the data obtained from a view defined by 'header' to spherical (Carrington) coordinates.
    '''

    from transforms import ToSpherical

    view = View.from_header(header).update(**kwargs)
    transform = ~ToSpherical() - view.to_carrington(**kwargs)

    dlon = 360 / ny

    if sine_lat:
        lat_spread = 2
    else:
        lat_spread = 180

    if staggered_lat:
        dlat = lat_spread / nx
    else:
        dlat = lat_spread / (nx - 1)

    grid = np.mgrid[-lat_spread / 2:lat_spread / 2 + dlat / 2:dlat, :360:dlon]

    if staggered_lat:
        grid = np.array([(grid[0][1:] + grid[0][:-1]) / 2, grid[1][1:]])

    if sine_lat:
        grid[0] = np.arcsin(grid[0].clip(-1,1)) * 180 / np.pi

    grid, alpha = transform(grid)
    data_ = map_coordinates(np.nan_to_num(data), grid, cval=np.nan)

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

    grid = view_new.grid()
    view = View.from_header(header)

    transform = (view_new.to_carrington() -
                 view.to_carrington(**kwargs))

    grid, alpha = transform(grid)
    data_ = map_coordinates(np.nan_to_num(data), grid, cval=np.nan)

    if correct_mu:
        data_ *= alpha

    if return_alpha:
        return data_, alpha
    else:
        return data_


def make_map(files, polar=False, correct_bias=False, **kwargs):
    '''
    Creates an averaged map by applying 'remap' routine to every dataset in files
    '''

    coverage = 0
    mean_data = 0
    mean_mu = 0
    var_mu = 0
    cov_data_mu = 0

    for file in files:
        mean_mu_ = mean_mu + 0.

        with fits.open(file) as hdul:
            header = hdul[0].header.copy()
            data = hdul[0].data.copy()

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


def make_map_(files, polar=False, **kwargs):
    '''
    Creates an averaged map by applying 'remap' routine to every dataset in files
    '''

    coverage = 0
    mean_data = 0
    mean_mu = 0
    var_mu = 0
    cov_data_mu = 0

    for file in files:
        with fits.open(file) as hdul:
            header = hdul[0].header.copy()
            data = hdul[0].data.copy()

        if polar:
            data, alpha = polar_view(data, header, return_alpha=True, correct_mu=False, **kwargs)
        else:
            data, alpha = remap(data, header, return_alpha=True, correct_mu=False, **kwargs)

        mu = 1 / alpha

        weight = (~np.isnan(data)).astype(float)
        coverage += np.nan_to_num(weight)

        mean_mu += np.nan_to_num((mu - mean_mu) * weight / coverage)
        mean_data += np.nan_to_num((data - mean_data) * weight / coverage)

        var_mu += np.nan_to_num((mu ** 2 - var_mu) * weight / coverage)
        cov_data_mu += np.nan_to_num((data * mu - cov_data_mu) * weight / coverage)

    mean_mu[coverage == 0] = np.nan
    mean_data[coverage == 0] = np.nan

    cov_data_mu[coverage == 0] = np.nan
    var_mu[coverage == 0] = np.nan
    coverage[coverage == 0] = np.nan

    return coverage, mean_mu, mean_data, var_mu, cov_data_mu
