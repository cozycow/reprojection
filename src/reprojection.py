import numpy as np
from interpolation import interp2d
from transforms import *


WSID = 360 / 25.38
WSYN = 360 / 27.2753
AU = 149597870691.
CLIGHT = 299792458
RSUN = 696000000.
A, B, C = 14.712, -2.396, -1.787  # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)
#P_CBS = [-893, 4134, -7347, 6963, -3352, 223]  # convective blue shift coefficients for HMI (Stief et al., A&A, 622, A34, 2019)
P_CBS = [-768, 2073, -1785, 462] # FDT fit


def func_mu(r, rsun_arc=0.):
    q = np.tan(rsun_arc / 3600 * np.pi / 180)
    return (r[2] - q) / np.sqrt(1 - 2 * r[2] * q + q ** 2)

def clip_mu(r, rsun_arc=0., thr=0.):
    return np.where(func_mu(r, rsun_arc=rsun_arc) > thr, 1, np.nan)


class View:
    def __init__(self, nx, ny, xc, yc, rsun, crota, crlt, crln, hgln=0.,
                 vr=0., vw=0., vn=0., rsun_arc=0., dsun=AU):
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
        self.hgln = hgln % 360
        self.vr = vr
        self.vw = vw
        self.vn = vn
        self.rsun_arc = rsun_arc
        self.dsun = dsun

    def update(self, increment=False, inplace=False, **kwargs):
        if not inplace:
            view_new = View(**self.__dict__)
        else:
            view_new = self

        for key, value in kwargs.items():
            if key in ['nx', 'ny', 'xc', 'yc', 'rsun', 'crota', 'crlt', 'crln', 'hgln',
                       'vr', 'vw', 'vn', 'rsun_arc', 'dsun']:
                if increment:
                    setattr(view_new, key, getattr(self, key) + value)
                else:
                    setattr(view_new, key, value)

        return view_new

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

        if 'RADIUS' in header:
            rsun = header['RADIUS']
        elif 'RSUN_ARC' in header:
            rsun = header['RSUN_ARC'] / header['CDELT1']
        else:
            rsun = header['RSUN_OBS'] / header['CDELT1']

        if 'CROTA' in header:
            crota = header['CROTA']
        elif 'CROTA2' in header:
            crota = header['CROTA2']
        else:
            crota = 0

        if 'HGLN_OBS' in header:
            hgln = header['HGLN_OBS']
        else:
            hgln = 0.

        if 'OBS_VR' in header:
            vr = header['OBS_VR']
        else:
            vr = 0.

        if 'OBS_VW' in header:
            vw = header['OBS_VW']
        else:
            vw = 0.

        if 'OBS_VN' in header:
            vn = header['OBS_VN']
        else:
            vn = 0.

        if 'RSUN_ARC' in header:
            rsun_arc = header['RSUN_ARC']
        elif 'RSUN_OBS' in header:
            rsun_arc = header['RSUN_OBS']
        else:
            rsun_arc = 0.

        if 'DSUN_OBS' in header:
            dsun = header['DSUN_OBS'] * np.cos(crlt * np.pi / 180)
        else:
            dsun = AU

        return cls(nx, ny, xc, yc, rsun, crota, crlt, crln, hgln, vr, vw, vn, rsun_arc, dsun)

    def get_transform(self, name='image', **kwargs):
        return getattr(self, 'to_' + name, Pipe)(**kwargs)

    def to_heliographic(self, correct_mu=False, mu_thr=0, origin='image', **kwargs):
        transform = (~self.get_transform(origin, **kwargs) +
                     Normalize((self.xc, self.yc), self.rsun) +
                     Make3d(self.rsun_arc / 3600))

        if correct_mu:
            transform += Filter(func_mu, rsun_arc=self.rsun_arc)

        transform += (Filter(clip_mu, rsun_arc=self.rsun_arc, thr=mu_thr) +
                      Rotate.z(self.crota * np.pi / 180))
        return transform

    def to_carrington(self, origin='image', **kwargs):
        tdel = (AU - self.dsun) / CLIGHT
        crln = self.crln + tdel * WSID / 24 / 3600

        transform = (~self.get_transform(origin, **kwargs) +
                     self.to_heliographic(**kwargs) -
                     Rotate.x(self.crlt * np.pi / 180) +
                     Rotate.y(crln * np.pi / 180))
        return transform

    def to_synoptic(self, stonyhurst=False, origin='image', **kwargs):
        tdel = (AU - self.dsun) / CLIGHT
        crln = self.crln + tdel * WSID / 24 / 3600

        if stonyhurst:
            crln0 = crln - self.hgln
            wsyn = WSYN
        else:
            crln0 = crln
            wsyn = WSID - self.vw / self.dsun / np.pi * 180 * 24 * 60 * 60

        transform = (~self.get_transform(origin, **kwargs) +
                     self.to_carrington(**kwargs) +
                     ToSpherical() +
                     ToSynoptic(crln0, Wsid=WSID, Wsyn=wsyn, A=A, B=B, C=C) -
                     ToSpherical())
        return transform

    def grid(self, origin='image', **kwargs):
        grid = np.mgrid[:self.nx, :self.ny].astype(np.float32)
        transform = self.get_transform(origin, **kwargs)
        grid, _ = transform(grid)
        return grid

    def reproject(self, image, view, grid=None, **kwargs):
        transform = self.to_carrington(**kwargs) - view.to_carrington(**kwargs)
        if grid is None:
            grid = self.grid(**kwargs)

        grid_, alpha = transform(grid)
        return interp2d(image, *grid_, **kwargs) * alpha

    def mu(self, *args, **kwargs):
        transform = self.to_heliographic(correct_mu=True, **kwargs)
        if len(args) > 0:
            _, alpha = transform(args)
        else:
            _, alpha = transform(self.grid(**kwargs))
        return alpha

    def velocity(self, **kwargs):
        xi, yi, zi = self.grid(origin='carrington', **kwargs)
        U = (A + B * yi ** 2 + C * yi ** 4) * RSUN * np.pi / 180 / 24 / 3600

        transform = self.to_heliographic(origin='carrington', **kwargs)
        v, _ = transform((zi * U, 0, -xi * U))
        vx, vy, vz = v[0] - self.vw, v[1] - self.vn, v[2] - self.vr
        xi, yi, zi = self.grid(origin='heliographic', **kwargs)

        q = np.tan(self.rsun_arc * np.pi / 180 / 3600)
        d = np.sqrt(1 - 2 * zi * q + q ** 2)
        V = (q * (xi * vx + yi * vy + zi * vz) - vz) / d
        return V


def reproject(data, header, header_new=None, **kwargs):
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
    return view_new.reproject(data, view, **kwargs)


def remap(data, header, dlon=1, dlat=1, **kwargs):
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
    transform = ~ToSpherical()  - view.to_carrington(**kwargs)

    grid = np.mgrid[-90:90 + dlat / 2:dlat, -180:180:dlon]
    grid, alpha = transform(grid)
    data = interp2d(data, *grid, **kwargs) * alpha
    return data

