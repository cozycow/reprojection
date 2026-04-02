import numpy as np
from interpolation import bilinear
from transforms import *


WSID = 360 / 25.38
WSYN = 360 / 27.2753
AU = 149597870691.
RSUN = 696000000.
A, B, C = 14.712, -2.396, -1.787  # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)
#P_CBS = [-893, 4134, -7347, 6963, -3352, 223]  # convective blue shift coefficients for HMI (Stief et al., A&A, 622, A34, 2019)
P_CBS = [-768, 2073, -1785, 462] # FDT fit

class View:
    def __init__(self, nx, ny, xc, yc, rsun, crota, crlt, crln, hgln=0., tdel=0.,
                 rsun_arc=0., vr=0., vw=0., vn=0., wsyn=0.):
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
        self.tdel = tdel
        self.rsun_arc = rsun_arc
        self.vr = vr
        self.vw = vw
        self.vn = vn
        self.wsyn = wsyn

    def update(self, increment=False, **kwargs):
        for key, value in kwargs.items():
            if key in ['nx', 'ny', 'xc', 'yc', 'rsun', 'crota', 'crlt', 'crln', 'hgln', 'tdel',
                       'rsun_arc', 'vr', 'vw', 'vn', 'wsyn']:
                if increment:
                    setattr(self, key, getattr(self, key) + value)
                else:
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

        if 'HGLN_OBS' in header:
            hgln = header['HGLN_OBS']
        else:
            hgln = 0.

        if 'EAR_TDEL' in header:
            tdel = header['EAR_TDEL']
        else:
            tdel = 0.

        if 'RSUN_ARC' in header:
            rsun_arc = header['RSUN_ARC']
        elif 'RSUN_OBS' in header:
            rsun_arc = header['RSUN_OBS']
        else:
            rsun_arc = 0.

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

        if 'DSUN_OBS' in header:
            wsyn = WSID - vw / header['DSUN_OBS'] / np.pi * 180 * 24 * 60 * 60
        else:
            wsyn = 0.

        return cls(nx, ny, xc, yc, rsun, crota, crlt, crln, hgln, tdel, rsun_arc, vr, vw, vn, wsyn)


    def helioproj(self, mu_thr=0):
        transform = (~Translate((self.xc, self.yc)) -
                     Scale(self.rsun) +
                     Expand(mu_thr) +
                     ToParaxial(theta=self.rsun_arc / 3600))

        transform_ = (~ToParaxial(theta=self.rsun_arc / 3600) -
                      Expand(mu_thr) +
                      Scale(self.rsun) +
                      Translate((self.xc, self.yc)))

        grid, _ = transform(self.grid)
        grid, _ = transform_(grid)
        return grid


    def to_spherical(self, correct_mu=False, correct_dr=False, stonyhurst=False, mu_thr=0, **kwargs):
        '''
        Constructs a transformation from image coordinates (in pixels) to Carrington coordinates (in degrees).

        :param correct_mu: bool, True is mu-correction needs to be applied.
        :param correct_dr: bool, True is differential rotation correction needs to be applied.
        :param mu_thr: Threshold for mu-correction.
        :return:
        '''

        transform = (~Translate((self.xc, self.yc)) -
                     Scale(self.rsun) +
                     Expand(thr=mu_thr) + ToParaxial(theta=self.rsun_arc / 3600))

        if correct_mu:
            transform += Filter(lambda r: -r[-1])

        transform += (~Rotate.z(self.crota * np.pi / 180) -
                     Rotate.y(self.crlt * np.pi / 180) +
                     Rotate.x((self.crln - self.tdel * WSID / 24 / 3600) * np.pi / 180) +
                     ToSpherical())

        if correct_dr:
            if stonyhurst:
                crln0 = self.crln - self.hgln
                wsyn = WSYN
            else:
                crln0 = self.crln
                wsyn = self.wsyn

            transform -= ToSynoptic(crln0, Wsid=WSID, Wsyn=wsyn, A=A, B=B, C=C)
        return transform

    @property
    def grid(self):
        return np.mgrid[:self.nx, :self.ny].astype(np.float32)

    def reproject(self, image, view, **kwargs):
        transform = self.to_spherical(**kwargs) - view.to_spherical(**kwargs)
        grid, alpha = transform(self.grid)
        return bilinear(image, *grid) * alpha

    def mu(self, thr=0):
        transform = (~Translate((self.xc, self.yc)) -
                     Scale(self.rsun) +
                     Expand(thr=thr))
        grid, _ = transform(self.grid)
        return -grid[2]

    def velocity(self, mu_thr=0, cbs=True, **kwargs):
        transform = (~Translate((self.xc, self.yc)) -
                     Scale(self.rsun) +
                     Expand(thr=mu_thr) -
                     Rotate.z(self.crota * np.pi / 180))

        grid, _ = transform(self.grid)
        xi, yi, zi = grid

        grid, _ = Rotate.y(-self.crlt * np.pi / 180)(grid)

        W = A + B * grid[0] ** 2 + C * grid[0] ** 4
        W = W * np.pi / 180 / 24 / 3600

        ew, _ = Rotate.y(-self.crlt * np.pi / 180)((1,0,0))

        Wx = W * ew[0]
        Wy = W * ew[1]
        Wz = W * ew[2]

        Vx = (Wy * zi - Wz * yi) * RSUN + self.vn
        Vy = (Wz * xi - Wx * zi) * RSUN + self.vw
        Vz = (Wx * yi - Wy * xi) * RSUN + self.vr

        q = np.tan(self.rsun_arc * np.pi / 180 / 3600)
        d = np.sqrt(xi ** 2 + yi ** 2 + (zi - 1 / q) ** 2)

        V = -(xi * Vx + yi * Vy + (zi - 1 / q) * Vz) / d

        if cbs:
            V += np.polyval(P_CBS, -zi)
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
    transform = ~view.to_spherical(**kwargs)

    grid = np.mgrid[-90:90 + dlat / 2:dlat, -180:180:dlon]
    grid, alpha = transform(grid)
    data = bilinear(data, *grid) * alpha
    return data

