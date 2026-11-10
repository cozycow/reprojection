import numpy as np
from astropy.io import fits
from view import View, Map


def reproject(data, header,
              header_new=None,
              kind='',
              mu_thr=0.1,
              **kwargs):

    view = View.from_header(header).update(minmu=mu_thr)

    if header_new is not None:
        view_new = View.from_header(header_new)

    elif kind == 'synoptic':
        nx, ny = 720, 1800
        view_new = Map(nx, ny, sine=True)

    elif kind == 'north':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        rsun = 0.45 * nx
        view_new = View(nx, ny, xc, yc, rsun, -90, 90, 0,
                        vw=np.inf, date=view.date, minmu=-1e-8, powmu=1.)

    elif kind == 'south':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        rsun = 0.45 * nx
        view_new = View(nx, ny, xc, yc, rsun, 90, -90, 0,
                        vw=np.inf, date=view.date, minmu=-1e-8, powmu=1.)

    elif kind == 'warp':
        nx, ny = 1024, 1024
        xc, yc = (nx - 1) / 2, (ny - 1) / 2
        rsun = 0.45 * nx
        view_new = View(nx, ny, xc, yc, rsun, 0, 0, 90,
                        rsun_deg=-45, vw=np.inf, date=view.date, powmu=1.)

    else:
        view_new = view.copy()

    view_new.update(inplace=True, **kwargs)
    data_ = view_new.reproject(data, view, **kwargs)

    return data_, view_new



def make_map(files,
             correct_crota=True,
             binning=1,
             correct_bias=False,
             return_variance=False,
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

        data = rebin(data, binning, update_header=header)

        x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1

        data, view = reproject(data, header, full_output=True, **kwargs)
        if view_out is None:
            view_out = view.copy()

        data, x, y, alpha = data
        data /= alpha
        z = 1 / alpha
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
    LLi = np.linalg.inv(LL)
    Q = LLi @ LF

    if correct_bias:
        GG[N == 0] = np.nan
        LG[N == 0] = np.nan
        GF[N == 0] = np.nan
        FF[N == 0] = np.nan
        N[N == 0] = np.nan

        R = LLi @ LG
        GL = np.swapaxes(LG, -2, -1)
        S = (FF - GF[:, :, -1:, :] ** 2 - Q ** 2 * (LL - GL[:, :, -1:, :] ** 2)) / LL / np.expand_dims(N - 1, (-1, -2))
        D = np.linalg.inv(np.nanmean((GG - GL @ R) / S, axis=(0, 1)))
        V = np.nanmean((GF - GL @ Q) / S, axis=(0, 1))
        P = D @ V
        dQ = R @ P
        Q -= dQ

    return np.squeeze(Q), view_out

