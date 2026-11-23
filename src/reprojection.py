import numpy as np
from astropy.io import fits
from view import Map, SunView
from diff_rot import *

ALPHA = 1.327124e20
class View(SunView):

    @property
    def keplerian_elements(self):
        vt = np.sqrt(self.vw ** 2 + self.vn ** 2)
        p = (self.dsun * vt) ** 2 / ALPHA
        v0 = np.sqrt(ALPHA / p)
        e = np.sqrt((vt / v0 - 1) ** 2 + (self.vr / v0) ** 2)
        a = p / (1 - e ** 2)
        theta = np.arctan2(self.vr, vt - v0) * 180 / np.pi % 360
        return a, e, theta

    @property
    def ecc_anomaly(self):
        a, e, theta = self.keplerian_elements
        cos_f = np.cos(theta * np.pi / 180)
        sin_f = np.sin(theta * np.pi / 180)
        return np.arctan2(np.sqrt(1 - e ** 2) * sin_f, e + cos_f) * 180 / np.pi

    @property
    def mean_anomaly(self):
        a, e, theta = self.keplerian_elements
        E = self.ecc_anomaly
        return E - e * np.sin(E * np.pi / 180) * 180 / np.pi

    @property
    def wsyn(self):
        a, e, theta = self.keplerian_elements
        wobs = np.sqrt(ALPHA / a ** 3) * 180 / np.pi * 24 * 3600
        return WSID - wobs

    @property
    def mean_crln(self):
        a, e, theta = self.keplerian_elements
        M = self.mean_anomaly

        axis_obs = (-self.vn, self.vw, 0)
        rot = Rotate.from_axis_angle(axis_obs, -theta) - Rotate.x(self.crlt) + Rotate.y(self.crln)
        r = rot(0, 0, 1)
        return (M + np.arctan2(r[0], r[2]) * 180 / np.pi) % 360

    def get_mu(self, x, y):
        transform = Normalize((self.xc, self.yc), (self.rsun, self.rsun)) + To3d()
        _, _, z = transform(x, y)
        q = np.sin(self.rsun_deg * np.pi / 180)
        return z * np.sqrt(1 - q ** 2) / np.sqrt(1 - (q * z) ** 2)

    @property
    def mu(self):
        return self.get_mu(*self.grid)

    def radialize(self, data):
        return data / self.mu

    def deradialize(self, data):
        return data * self.mu


def reproject(data, header,
              header_new=None,
              kind='',
              minmu=0.1,
              car_rot=None,
              correct_dr=False,
              **kwargs):

    view = View.from_header(header)

    if header_new is not None:
        view_new = View.from_header(header_new)
    elif kind == 'synoptic':
        nx, ny = 720, 1800
        view_new = Map(nx, ny, sine=True)
    else:
        view_new = view.copy()

    if kind == 'north':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        rsun = 0.45 * nx

        view_new.update(inplace=True, nx=nx, ny=ny, xc=xc, yc=yc, rsun=rsun, rsun_deg=0,
                        crota=-90, crlt=90, crln=0, minmu=-1e-8)

    elif kind == 'south':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        rsun = 0.45 * nx

        view_new.update(inplace=True, nx=nx, ny=ny, xc=xc, yc=yc, rsun=rsun, rsun_deg=0,
                        crota=90, crlt=-90, crln=0, minmu=-1e-8)

    elif kind == 'warp':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        rsun = 0.45 * nx

        view_new.update(inplace=True, nx=nx, ny=ny, xc=xc, yc=yc, rsun=rsun,
                        crota=0, crlt=0, crln=0, minmu=-1e-8,
                        rsun_deg=-45)

    Q = []
    if correct_dr:
        Q += [DiffRotate(view.date, car_rot, wsyn=view.wsyn, epoch='20260710T030009', crln0=207.18, **kwargs)]

    view_new.update(inplace=True, **kwargs)
    data_ = view_new.reproject(view.radialize(data), view.update(minmu=minmu), *Q, **kwargs)

    if kind == 'north' or kind == 'south':
        data_ = view_new.deradialize(data_)

    return data_, view_new



def make_map(files,
             correct_crota=True,
             binning=1,
             correct_bias=False,
             full_output=False,
             **kwargs):
    '''
    Creates an averaged map by applying 'remap' routine to every dataset in files
    '''

    from utils import rebin

    N, LF, LL = 0, 0, 0
    LL, GG, LG, FF, GF = 0, 0, 0, 0, 0

    view_out = None

    for file in files:
        with fits.open(file) as hdul:
            #print(file)

            if 'hmi' in file:
                data = hdul[1].data.copy()
                header = hdul[1].header.copy()
            else: # 'fdt'
                data = hdul[0].data.copy()
                header = hdul[0].header.copy()

                if correct_crota:
                    header['CROTA'] = header['CROTA'] + 0.25 ### crota correction

        #rsun = header['RSUN_ARC'] / header['CDELT1']
        #binning = int(np.ceil(rsun / 450).clip(1))

        data = rebin(data, binning, update_header=header)

        x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1

        data, view = reproject(data, header, full_output=True, **kwargs)
        if view_out is None:
            view_out = view.copy()

        data, x, y = data
        z = View.from_header(header).get_mu(x, y)
        data *= z
        n = (~np.isnan(data)).astype(float)

        N += np.nan_to_num(n)
        w = np.expand_dims(n / N, (-2, -1))

        f = np.expand_dims(data, (-2, -1))
        l = np.expand_dims(z, (-2, -1))
        ll = np.swapaxes(l, -2, -1) @ l
        lf = np.swapaxes(l, -2, -1) @ f

        LF += np.nan_to_num((lf - LF) * w)
        LL += np.nan_to_num((ll - LL) * w)

        if correct_bias:
            x, y = (x + x0 - 1024) / 1024, (y + y0 - 1024) / 1024
            g = np.moveaxis(np.array([[x, y, n]]), (0, 1), (-2, -1))
        else:
            g = np.moveaxis(np.array([[n]]), (0, 1), (-2, -1))

        gg = np.swapaxes(g, -2, -1) @ g
        lg = np.swapaxes(l, -2, -1) @ g
        gf = np.swapaxes(g, -2, -1) @ f
        ff = np.swapaxes(f, -2, -1) @ f

        GG += np.nan_to_num((gg - GG) * w)
        LG += np.nan_to_num((lg - LG) * w)
        GF += np.nan_to_num((gf - GF) * w)
        FF += np.nan_to_num((ff - FF) * w)

    LL[N == 0] = np.nan
    LF[N == 0] = np.nan
    GG[N == 0] = np.nan
    LG[N == 0] = np.nan
    GF[N == 0] = np.nan
    FF[N == 0] = np.nan
    N[N == 0] = np.nan

    GL = np.swapaxes(LG, -2, -1)
    LLi = np.linalg.inv(LL)
    Q = LLi @ LF
    S = (FF - GF[:, :, -1:, :] ** 2 - Q ** 2 * (LL - GL[:, :, -1:, :] ** 2)) / LL / np.expand_dims(N - 1, (-1, -2))

    if correct_bias:
        R = LLi @ LG
        D = np.linalg.inv(np.nanmean((GG - GL @ R) / S, axis=(0, 1)))
        V = np.nanmean((GF - GL @ Q) / S, axis=(0, 1))
        P = D @ V
        dQ = R @ P
        Q -= dQ

    out = np.squeeze(Q)
    if full_output:
        out = np.array([out, np.squeeze(S), N])

    return out, view_out

