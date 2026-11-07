from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import map_coordinates
from transforms import *
from diff_rot import *


WSID = 360 / 25.38
AU = 149597870691.
CLIGHT = 299792458
RSUN = 696e8


def get_jdn(date):
    from datetime import datetime

    if isinstance(date, str):
        return get_jdn(datetime.fromisoformat(date))

    date0 = datetime(2000, 1, 1, 12, 0, 0)
    jd0 = 2451545.

    jd = jd0 + (date - date0).total_seconds() / 24 / 60 / 60
    return jd


class ViewBase(ABC):

    def __init__(self, *, nx, ny, **kwargs):
        self.nx = nx
        self.ny = ny

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def to_carrington(self, **kwargs):
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
    def grid(self):
        grid = tuple(np.mgrid[:self.nx, :self.ny].astype(np.float32))
        return grid

    def reproject(self, image, view, **kwargs):
        transform = self.to_carrington(**kwargs) - view.to_carrington(**kwargs)
        grid_, alpha = transform(self.grid)

        image_ = map_coordinates(np.nan_to_num(image), grid_, cval=np.nan)
        return image_, alpha


class Map(ViewBase):
    def __init__(self, nx, ny, sine=False):
        super().__init__(nx=nx, ny=ny, sine=sine)

    def to_carrington(self, **kwargs):
        transform = (Normalize(((self.nx - 1) / 2, 0), ((self.nx - 1) / 180, self.ny / 360)) -
                     ToSpherical(self.sine))
        return transform


class View(ViewBase):
    def __init__(self, nx, ny, xc, yc, rsun, crota, crlt, crln, hgln=0.,
                 vr=0., vw=0., vn=0., rsun_deg=0., dsun=AU, jdn=0., minmu=-np.inf, powmu=0.):
        '''
        A WCS information container.
        '''
        super().__init__(nx=nx, ny=ny, xc=xc, yc=yc, rsun=rsun,
                         crota=crota, crlt=crlt, crln=crln % 360, hgln=hgln % 360,
                         vr=vr, vw=vw, vn=vn, rsun_deg=rsun_deg, dsun=dsun,
                         jdn=jdn, minmu=minmu, powmu=powmu)

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
            jdn = get_jdn(header['DATE-OBS'])
        else:
            jdn = 0.

        return cls(nx, ny, xc, yc, rsun, crota, crlt, crln, hgln, vr, vw, vn, rsun_deg, dsun, jdn, 0., 1.)

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

        header_new['CDELT1'] = self.rsun_deg * 3600 / self.rsun
        header_new['CDELT2'] = self.rsun_deg * 3600 / self.rsun
        return header_new

    def to_heliographic(self, **kwargs):

        def mu_filter(r, minmu=0., theta=0.):
            x, y, z = r
            q = np.sin(theta * np.pi / 180)
            mu = (z - q) / np.sqrt(q ** 2 - 2 * q * z + 1)
            t = np.where(mu < minmu)
            x[t], y[t], z[t], mu[t] = np.nan, np.nan, np.nan, np.nan
            return mu

        transform = (Normalize((self.xc, self.yc), (self.rsun, self.rsun)) +
                     Make3d(self.rsun_deg) +
                     Filter(mu_filter, self.powmu, minmu=self.minmu, theta=self.rsun_deg) +
                     Rotate.z(self.crota))

        return transform

    def to_carrington(self, correct_dr=False, **kwargs):
        tdel = (AU - self.dsun) / CLIGHT / 24 / 3600
        crln = self.crln + tdel * WSID

        rot = ~Rotate.x(self.crlt) + Rotate.y(crln)
        transform = (self.to_heliographic(**kwargs) +
                     rot)

        if correct_dr:
            #transform -= DiffRotate(self.jdn - tdel, diff_rot)
            transform -= DiffRotate(1, diff_rot2,
                                    dsun_obs=self.dsun, obs_vn=self.vn, obs_vw=self.vw, obs_vr=self.vr,
                                    crln_obs=crln, crlt_obs=self.crlt)

        return transform



