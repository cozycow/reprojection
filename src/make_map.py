import numpy as np
from astropy.io import fits
from reprojection import *


def make_map(files,
             correct_crota=True,
             correct_bias=False,
             full_output=False,
             rebin_to_rsun=None,#400,
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

        if rebin_to_rsun is not None:
            rsun = header['RSUN_ARC'] / header['CDELT1']
            binning = max(rsun / rebin_to_rsun, 1.)
            data = rebin(data, binning, update_header=header)
        else:
            binning = 1
            data = data

        x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1

        data, view = reproject(data, header, full_output=True, **kwargs)
        if view_out is None:
            view_out = view.copy()

        cdn = (datetime.fromisoformat(header['DATE-OBS']) -
               datetime.fromisoformat('20260901T000000')).total_seconds() / 24 / 3600

        data, x, y, z = data
        x, y = ((x - 0.5) * binning + 0.5 + x0 - 1024) / 1024, ((y - 0.5) * binning + 0.5 + y0 - 1024) / 1024
        data *= z
        #data -= x * (-1.81015285) + y * (-0.06079465) -0.42041103
        #data -= x * (-0.3486008) + y * (-0.23286657) -0.23242867

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
            #g = np.moveaxis(np.array([[x ** 2, x * y, y ** 2, x, y, n]]), (0, 1), (-2, -1))
            #g = np.moveaxis(np.array([[x, y, n * cdn, n]]), (0, 1), (-2, -1))
            g = np.moveaxis(np.array([[x, y, n]]), (0, 1), (-2, -1))
            #g = np.moveaxis(np.array([[n]]), (0, 1), (-2, -1))
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

    N[:50] = 0
    N[-50:] = 0
    N[128:-128] = 0

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
        D = np.linalg.inv(np.nanmean((GG - GL @ R), axis=(0, 1)))
        V = np.nanmean((GF - GL @ Q), axis=(0, 1))
        P = D @ V
        dQ = R @ P
        Q -= dQ

        print(P)

    out = np.squeeze(Q)
    if full_output:
        out = np.array([out, np.squeeze(S), N])

    return out, view_out