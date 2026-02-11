import numpy as np
from interpolation import bilinear
from transforms import *


class View:
    def __init__(self, nx, ny, xc, yc, rsun, crota, crlt, crln, x0=0, y0=0, ww=0.985):
        '''
        A WCS information container.

        :param nx:
        :param ny:
        :param xc:
        :param yc:
        :param rsun:
        :param crota:
        :param crlt:
        :param crln:
        :param x0:
        :param y0:
        :param ww:
        '''

        self.nx = nx
        self.ny = ny
        self.xc = xc
        self.yc = yc
        self.rsun = rsun
        self.crota = crota
        self.crlt = crlt
        self.crln = crln % 360
        self.x0 = x0
        self.y0 = y0
        self.ww = ww

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in ['nx', 'ny', 'xc', 'yc', 'rsun', 'crota', 'crlt', 'crln', 'x0', 'y0', 'ww']:
                setattr(self, key, value)
        return self

    @classmethod
    def from_header(cls, header):
        '''
        Reads WCS information from header.

        :param header:
        :return:
        '''

        nx, ny = header['NAXIS2'], header['NAXIS1']
        xc, yc = header['CRPIX2'] - 1, header['CRPIX1'] - 1
        crlt, crln = header['CRLT_OBS'], header['CRLN_OBS']

        if 'PXBEG2' in header:
            x0 = header['PXBEG2'] - 1
        else:
            x0 = 0
        if 'PXBEG1' in header:
            y0 = header['PXBEG1'] - 1
        else:
            y0 = 0

        if 'RADIUS' in header:
            rsun = header['RADIUS']
        elif 'RSUN_ARC' in header:
            rsun = header['RSUN_ARC'] / header['CDELT1']
        else:
            rsun = header['RSUN_OBS'] / header['CDELT1']

        if 'CROTA' in header:
            crota = header['CROTA']
        else:
            crota = header['CROTA2']

        if 'OBS_VW' in header:
            vw = header['OBS_VW'] ## Westward velocity
            d_sun = header['DSUN_OBS'] ## Distance to the Sun
            ww = vw / d_sun / np.pi * 180 * 24 * 60 * 60  ## Westward rotation rate in degrees per day
        else:
            ww = 0.985

        return cls(nx, ny, xc, yc, rsun, crota, crlt, crln, x0, y0, ww)

    def to_spherical(self, correct_mu=False, correct_dr=False, mu_thr=0):
        '''
        Constructs a transformation from image coordinates (in pixels) to Carrington coordinates (in degrees).

        :param correct_mu: bool, True is mu-correction needs to be applied.
        :param correct_dr: bool, True is differential rotation correction needs to be applied.
        :param mu_thr: Threshold for mu-correction.
        :return:
        '''

        transform = (~Translate((self.xc, self.yc)) -
                     Scale(self.rsun) +
                     Expand(thr=mu_thr))

        if correct_mu:
            transform += Filter(lambda r: r[-1])

        transform += (~Rotate.z(self.crota * np.pi / 180) +
                     Rotate.y(self.crlt * np.pi / 180) -
                     Rotate.x(self.crln * np.pi / 180) +
                     ToSpherical())

        if correct_dr:
            Wsid = 14.184
            Wsyn = Wsid - self.ww
            transform -= ToSynoptic(self.crln, Wsid=Wsid, Wsyn=Wsyn)

        return transform


def reproject(data, header, header_new=None, correct_mu=False, mu_thr=0, **kwargs):
    '''
    Reprojects the data obtained from a view defined by 'header' to a view defined by 'header_new'.

    :param data:
    :param header:
    :param header_new:
    :param correct_mu:
    :param mu_thr:
    :param kwargs:
    :return:
    '''


    if header_new is None:
        header_ = header
    else:
        header_ = header_new

    view = View.from_header(header)
    view_new = View.from_header(header_).update(**kwargs)
    transform = view_new.to_spherical(correct_mu=correct_mu) - view.to_spherical(correct_mu=correct_mu, mu_thr=mu_thr)

    grid = np.mgrid[:view_new.nx, :view_new.ny]
    grid, alpha = transform(grid)
    data = bilinear(data, *grid) * alpha
    return data


def remap(data, header, dlon=1, dlat=1, correct_mu=False, mu_thr=0, **kwargs):
    '''
    Remaps the data obtained from a view defined by 'header' to spherical (Carrington) coordinates.

    :param data:
    :param header:
    :param dlon:
    :param dlat:
    :param correct_mu:
    :param mu_thr:
    :param kwargs:
    :return:
    '''

    view = View.from_header(header).update(**kwargs)
    transform = ~view.to_spherical(correct_mu=correct_mu, mu_thr=mu_thr)

    grid = np.mgrid[-90:90 + dlat / 2:dlat, -180:180:dlon]
    grid, alpha = transform(grid)
    data = bilinear(data, *grid) * alpha
    return data

