import numpy as np
from scipy.ndimage import map_coordinates
from transforms import *


WSID = 360 / 25.38
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

    def update(self, increment=False, multiply=False, inplace=False, **kwargs):
        if not inplace:
            view_new = self.copy()
        else:
            view_new = self

        for key, value in kwargs.items():
            if key in ['nx', 'ny', 'xc', 'yc', 'rsun', 'crota', 'crlt', 'crln', 'hgln',
                       'vr', 'vw', 'vn', 'rsun_arc', 'dsun']:
                if increment:
                    setattr(view_new, key, getattr(self, key) + value)
                elif multiply:
                    setattr(view_new, key, getattr(self, key) * value)
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


    def update_header(self, header, inplace=False):
        '''
        Updates WCS information in header

        :param header:
        :return:
        '''


        if inplace:
            header_new = header
        else:
            header_new = header.copy()

        header_new['NAXIS2'] = self.nx
        header_new['NAXIS1'] = self.ny
        header_new['CRPIX2'] = self.xc
        header_new['CRPIX1'] = self.yc

        if 'RSUN_ARC' in header_new:
            header_new['RSUN_ARC'] = self.rsun_arc
        elif 'RSUN_OBS' in header_new:
            header_new['RSUN_OBS'] = self.rsun_arc

        if 'CROTA' in header_new:
            header_new['CROTA'] = self.crota
        elif 'CROTA2' in header_new:
            header_new['CROTA2'] = self.crota

        header_new['CRLT_OBS'] = self.crlt
        header_new['CRLN_OBS'] = self.crln

        if 'HGLN_OBS' in header_new:
            header_new['HGLN_OBS'] = self.hgln

        if 'RADIUS' in header_new:
            header_new['RADIUS'] = self.rsun

        if 'DSUN_OBS' in header_new:
            header_new['DSUN_OBS'] = self.dsun

        header_new['CDELT1'] = self.rsun_arc / self.rsun
        header_new['CDELT2'] = self.rsun_arc / self.rsun

        return header_new


    @property
    def wsyn(self):
        return WSID - self.vw / self.dsun / np.cos(self.crlt * np.pi / 180) / np.pi * 180 * 24 * 60 * 60

    def get_transform(self, name='image', **kwargs):
        return getattr(self, 'to_' + name, Pipe)(**kwargs)

    def to_heliographic(self, minmu=0, maxmu=1, origin='image', **kwargs):
        transform = (~self.get_transform(origin, **kwargs) +
                     Normalize((self.xc, self.yc), self.rsun) +
                     Make3d(self.rsun_arc / 3600, minmu=minmu, maxmu=maxmu))

        transform += Rotate.z(self.crota * np.pi / 180)
        return transform

    def to_carrington(self, origin='image', correct_dr=False, delta_t=1, stonyhurst=False, **kwargs):
        tdel = (AU - self.dsun) / CLIGHT
        crln = self.crln + tdel * WSID / 24 / 3600

        if stonyhurst:
            crln0 = crln - self.hgln
        else:
            crln0 = crln

        transform = (~self.get_transform(origin, **kwargs) +
                     self.to_heliographic(**kwargs) -
                     Rotate.x(self.crlt * np.pi / 180) +
                     Rotate.y(crln * np.pi / 180))

        if correct_dr:
            transform += (ToSpherical() +
                         ToSynoptic(crln0, A, B, C, WSID, self.wsyn, delta_t=delta_t) -
                         ToSpherical())
        return transform

    def grid(self, origin='image', **kwargs):
        grid = tuple(np.mgrid[:self.nx, :self.ny].astype(np.float32))
        transform = self.get_transform(origin, **kwargs)
        grid, _ = transform(grid)
        return grid

    def reproject(self, image, view, **kwargs):
        transform = self.to_carrington() - view.to_carrington(**kwargs)
        grid = self.grid(**kwargs)
        grid_, alpha = transform(grid)

        image_ = map_coordinates(np.nan_to_num(image), grid_, cval=np.nan)
        return image_, alpha

    def remap(self, image, nx=720, ny=1800, sine_lat=False, **kwargs):
        transform = self.to_carrington(**kwargs) + ToSpherical()

        dlat, dlon = 180 / (nx - 1), 360 / ny
        if sine_lat:
            sine_transform = Custom(lambda r, alpha: ((np.sin(r[0] * np.pi / 180) * 90,) + r[1:], alpha),
                                    lambda r, alpha: ((np.arcsin(r[0] / 90) * 180 / np.pi,) + r[1:], alpha))
            transform = transform + sine_transform

        grid = tuple(np.mgrid[-90:90 + dlat / 2:dlat, :360:dlon])
        grid, alpha = (~transform)(grid)
        image_ = map_coordinates(np.nan_to_num(image), grid, cval=np.nan)
        return image_, alpha
