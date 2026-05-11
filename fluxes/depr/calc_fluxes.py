import numpy as np
from astropy.io import fits
from reprojection import *
from utils import *
import pandas as pd


s = np.load('/home/ulyanov/data/solo/phi/distortion/fdt/distortion_cor.npz')
xu, yu = s['xu'], s['yu']

df = pd.read_csv('/home/ulyanov/data/solo/phi/wcs/fdt/disk_centers_cor.csv', skipinitialspace=True).dropna()
dids = df['did'].to_numpy()
xu_sun, yu_sun, ru_sun = df['xu_sun'].to_numpy(), df['yu_sun'].to_numpy(), df['ru_sun'].to_numpy()


def calc_fluxes_(file, dsine=0.01, mu_thr=0.1):
    with fits.open(file) as hdul:
        header = hdul[0].header.copy()
        data = hdul[0].data.copy()

    did = int(file.split('_')[-1].split('.')[0])

    view = View.from_header(header)
    view.update(xc=xu_sun[dids == did][0], yc=yu_sun[dids == did][0], crota=view.crota + 0.25, rsun=ru_sun[dids == did][0], inplace=True)

    transform = view.to_carrington(correct_mu=True, mu_thr=mu_thr) + ToSpherical()
    grid, mu = transform(crop_grid(xu, yu, header))

    Br = data / mu

    sine = np.arange(-1,1 + dsine / 2,dsine)
    lati = np.arcsin(sine.clip(-1,1)) * 180 / np.pi

    sine_map = np.sin(grid[0] * np.pi / 180) // dsine

    weight = np.zeros_like(lati[:-1]).astype(np.float32)
    flux_density = np.zeros_like(lati[:-1]).astype(np.float32)

    for i, sine_ in enumerate(sine):
        t = (sine_map == int(sine_ / dsine))
        nt = np.sum(t)

        if nt > 0:
            Br_ = Br[t]
            W_ = mu[t] ** 4 / mu[t]

            weight[i] = np.nansum(W_)
            flux_density[i] = np.nansum(W_ * Br_) / weight[i]

    return lati, weight, flux_density


def calc_fluxes(files, **kwargs):
    x, mean, w_sum, w_sum2, S = 0., 0., 0. ,0., 0.

    for file in files:
        x, w, y = calc_fluxes_(file, **kwargs)

        w_sum += w
        w_sum2 += w ** 2
        mean_old = mean + 0.
        mean += np.nan_to_num((w / w_sum) * (y - mean))
        S += np.nan_to_num(w * (y - mean_old) * (y - mean))

    S *= w_sum2 / w_sum ** 3
    return x, mean, S