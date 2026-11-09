from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import map_coordinates
from transforms import *
from diff_rot import *


WSID = 360 / 25.38
ALPHA = 1.327124e20
AU = 149597870691.
CLIGHT = 299792458
RSUN = 696e8


class ViewBase(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    def copy(self):
        return type(self)(**self.__dict__)

    def update(self, increment=False, multiply=False, inplace=False, **kwargs):
        if not inplace:
            view_new = self.copy()
        else:
            view_new = self

        for key, value in kwargs.items():
            if key in self.__dict__:
                if increment:
                    setattr(view_new, key, getattr(self, key) + value)
                elif multiply:
                    setattr(view_new, key, getattr(self, key) * value)
                else:
                    setattr(view_new, key, value)

        return view_new

    @property
    @abstractmethod
    def grid(self):
        pass

    @abstractmethod
    def transform(self, **kwargs):
        pass

    def reproject(self, image, view, return_alpha=False, **kwargs):
        transform = self.transform(**kwargs) - view.transform(**kwargs)
        xi, yi, alpha = transform(*self.grid, 1)

        image_ = map_coordinates(np.nan_to_num(image), (xi, yi), cval=np.nan) * alpha

        if return_alpha:
            image_ = np.array([image_, alpha])

        return image_


class Map(ViewBase):
    def __init__(self, nx, ny, sine=False):
        self.nx = nx
        self.ny = ny
        self.sine = sine

    def transform(self, **kwargs):
        transform = (Normalize(((self.nx - 1) / 2, 0), ((self.nx - 1) / 180, self.ny / 360)) -
                     ToSpherical(self.sine))
        return transform

    @property
    def grid(self):
        grid = tuple(np.mgrid[:self.nx, :self.ny].astype(np.float32))
        return grid


class View(ViewBase):
    def __init__(self, nx, ny, xc, yc, rsun, crota, crlt, crln, hgln=0.,
                 vr=0., vw=0., vn=0., rsun_deg=0., dsun=AU, date='', minmu=-np.inf, powmu=0.):
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
        self.rsun_deg = rsun_deg
        self.dsun = dsun
        self.date = date
        self.minmu = minmu
        self.powmu = powmu


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
            rsun_deg = header['RSUN_ARC'] / 3600
        elif 'RSUN_OBS' in header:
            rsun_deg = header['RSUN_OBS'] / 3600
        else:
            rsun_deg = 0.

        if 'DSUN_OBS' in header:
            dsun = header['DSUN_OBS']
        else:
            dsun = AU

        if 'DATE-OBS' in header:
            date = header['DATE-OBS']
        else:
            date = ''

        return cls(nx, ny, xc, yc, rsun, crota, crlt, crln, hgln, vr, vw, vn, rsun_deg, dsun, date, 0., 1.)

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
            header_new['RSUN_ARC'] = self.rsun_deg * 3600
        elif 'RSUN_OBS' in header_new:
            header_new['RSUN_OBS'] = self.rsun_deg * 3600

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

        if 'DATE-OBS' in header_new:
            header_new['DATE-OBS'] = self.date

        header_new['CDELT1'] = self.rsun_deg * 3600 / self.rsun
        header_new['CDELT2'] = self.rsun_deg * 3600 / self.rsun
        return header_new

    @property
    def grid(self):
        grid = tuple(np.mgrid[:self.nx, :self.ny].astype(np.float32))
        return grid

    @property
    def cdn(self):
        from datetime import datetime

        date = datetime.fromisoformat(self.date)
        date0 = datetime(1853, 11, 9, 0, 0, 0)

        return (date - date0).total_seconds() / 24 / 60 / 60

    @property
    def kepler(self):
        vt = np.sqrt(self.vw ** 2 + self.vn ** 2)
        p = (self.dsun * vt) ** 2 / ALPHA
        v0 = np.sqrt(ALPHA / p)
        e = np.sqrt((vt / v0 - 1) ** 2 + (self.vr / v0) ** 2)
        a = p / (1 - e ** 2)
        theta = np.arctan2(self.vr, vt - v0) * 180 / np.pi % 360
        return a, e, theta

    @property
    def ecc_anomaly(self):
        a, e, theta = self.kepler
        cos_f = np.cos(theta * np.pi / 180)
        sin_f = np.sin(theta * np.pi / 180)
        return np.arctan2(np.sqrt(1 - e ** 2) * sin_f, e + cos_f) * 180 / np.pi

    @property
    def mean_anomaly(self):
        a, e, theta = self.kepler
        E = self.ecc_anomaly
        return E - e * np.sin(E * np.pi / 180) * 180 / np.pi

    @property
    def wsyn(self):
        a, e, theta = self.kepler
        wobs = np.sqrt(ALPHA / a ** 3) * 180 / np.pi * 24 * 3600
        return WSID - wobs

    @property
    def mean_crln(self):
        a, e, theta = self.kepler
        M = self.mean_anomaly
        rot = ~Rotate.x(self.crlt) + Rotate.y(self.crln)
        r_obs = rot(0, 0, 1)
        axis_obs = rot(-self.vn, self.vw, 0)
        r_ = Rotate.from_axis_angle(axis_obs, -theta)(*r_obs)
        return M + np.arctan2(r_[0], r_[2]) * 180 / np.pi

    def to_heliographic(self, **kwargs):

        transform = (Normalize((self.xc, self.yc), (self.rsun, self.rsun)) +
                     To3d() +
                     ToParaxial(self.rsun_deg) +
                     MuFilter(self.minmu, self.rsun_deg, self.powmu) +
                     Rotate.z(self.crota))

        return transform

    def transform(self, correct_dr=False, **kwargs):
        tdel = (AU - self.dsun) / CLIGHT / 24 / 3600
        crln = self.crln + tdel * WSID

        rot = ~Rotate.x(self.crlt) + Rotate.y(crln)
        transform = (self.to_heliographic(**kwargs) +
                     rot)

        if correct_dr:
            #transform -= DiffRotate(self.cdn - tdel, diff_rot)
            transform -= DiffRotate(1, diff_rot2,
                                    delta=self.mean_crln, wsyn=self.wsyn)

        return transform
