import numpy as np
from scipy.ndimage import map_coordinates
from transforms import *


WSID = 360 / 25.38
WSYN = 360 / 27.2753
AU = 149597870691.
CLIGHT = 299792458
RSUN = 696e8
A, B, C = 14.712, -2.396, -1.787  # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)


class View:
    def __init__(self, nx, ny, xc, yc, rsun, crota, crlt, crln, hgln=0.,
                 vr=0., vw=0., vn=0., rsun_arc=0., dsun=AU):
        '''
        A WCS information container.
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


    def copy(self):
        return View(**self.__dict__)

    def update(self, increment=False, inplace=False, **kwargs):
        if not inplace:
            view_new = self.copy()
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
            dsun = header['DSUN_OBS']
        else:
            dsun = AU

        return cls(nx, ny, xc, yc, rsun, crota, crlt, crln, hgln, vr, vw, vn, rsun_arc, dsun)

    def get_transform(self, name='image', **kwargs):
        return getattr(self, 'to_' + name, Pipe)(**kwargs)

    def to_heliographic(self, mu_thr=0, origin='image', **kwargs):
        transform = (~self.get_transform(origin, **kwargs) +
                     Normalize((self.xc, self.yc), self.rsun) +
                     Make3d(self.rsun_arc / 3600, mu_thr=mu_thr))

        transform += Rotate.z(self.crota * np.pi / 180)
        return transform

    def to_carrington(self, origin='image', correct_dr=False, delta_lon=180, **kwargs):
        tdel = (AU - self.dsun) / CLIGHT
        crln = self.crln + tdel * WSID / 24 / 3600

        transform = (~self.get_transform(origin, **kwargs) +
                     self.to_heliographic(**kwargs) -
                     Rotate.x(self.crlt * np.pi / 180) +
                     Rotate.y(crln * np.pi / 180))

        if correct_dr:
            wsyn = WSID - self.vw / self.dsun / np.cos(self.crlt * np.pi / 180) / np.pi * 180 * 24 * 60 * 60
            transform += (ToSpherical() +
                         ToSynoptic(crln, Wsid=WSID, Wsyn=wsyn, A=A, B=B, C=C, delta_lon=delta_lon) -
                         ToSpherical())
        return transform

    def grid(self, origin='image', **kwargs):
        grid = np.mgrid[:self.nx, :self.ny].astype(np.float32)
        transform = self.get_transform(origin, **kwargs)
        grid, _ = transform(grid)
        return grid

    def reproject(self, image, view, correct_mu=False, **kwargs):
        transform = self.to_carrington(**kwargs) - view.to_carrington(**kwargs)
        grid = self.grid(**kwargs)
        grid_, alpha = transform(grid)

        image_ = map_coordinates(np.nan_to_num(image), grid_, cval=np.nan)
        if correct_mu:
            image_ *= alpha
        return image_
