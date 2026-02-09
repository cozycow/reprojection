import numpy as np


def reproject(data, header, header_new=None, correct_mu=False, mu_thr=0, **kwargs):
    from transforms import View
    from interpolation import bilinear

    if header_new is None:
        header_ = header
    else:
        header_ = header_new

    view = View.from_header(header)
    view_new = View.from_header(header_).update(**kwargs)

    transform = (~view_new.to_spherical(correct_mu=correct_mu, **kwargs) +
                 view.to_spherical(correct_mu=correct_mu, mu_thr=mu_thr, **kwargs))

    grid = np.mgrid[:view_new.nx, :view_new.ny]
    grid, alpha = transform(grid)
    data = bilinear(data, *grid) * alpha
    return data


def remap(data, header, correct_mu=False, dlon=1, dlat=1, mu_thr=0, **kwargs):
    from transforms import View
    from interpolation import bilinear

    view = View.from_header(header).update(**kwargs)
    transform = view.to_spherical(correct_mu=correct_mu, mu_thr=mu_thr)
    grid = np.mgrid[-90:90 + dlat/2:dlat, -180:180:dlon]
    grid, alpha = transform(grid)
    data = bilinear(data, *grid) * alpha
    return data

