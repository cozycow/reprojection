import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sunpy.visualization.colormaps as cm
from matplotlib import colormaps
from transforms import ToSpherical
from view import View


def show_data(data, view, title='', label='', figsize=(10,10), cmap='hmimag', vmin=-200, vmax=200,
              grid_alpha=0.5, grid_color='black', text_alpha=1, text_color='black',
              to_file=None):

    transform = ~(view.to_carrington(mu_thr=-1e-8) + ToSpherical())
    _, _, mu = view.grid(origin='heliographic')
    data_ = data.copy()
    data_[np.isnan(mu)] = np.nan

    grid = np.mgrid[-90:90.5:1,0:360.5:1]
    grid, _ = transform(grid)
    grid = np.array(grid)

    meridians = grid[:,:,::15]
    parallels = np.transpose(grid[:,::15,:], (0,2,1))

    if to_file is not None:
        plt.ioff()

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(data_, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(meridians[1][:,1:-1], meridians[0][:,1:-1], color=grid_color, ls='--', lw=0.5, alpha=grid_alpha)
    ax.plot(meridians[1][:,0], meridians[0][:,0], color=grid_color, ls='-', lw=0.5, alpha=grid_alpha) # central meridian

    ax.plot(parallels[1][:,:6], parallels[0][:,:6], color=grid_color, ls='--', lw=0.5, alpha=grid_alpha)
    ax.plot(parallels[1][:,6], parallels[0][:,6], color=grid_color, ls='-', lw=0.5, alpha=grid_alpha) # equator
    ax.plot(parallels[1][:,7:], parallels[0][:,7:], color=grid_color, ls='--', lw=0.5, alpha=grid_alpha)

    ax.add_patch(Circle((view.yc, view.xc), radius=view.rsun, color='black', ls='-', lw=0.5, alpha=1, fill=False)) # limb

    for i in range(0,360,15):
        ax.annotate(i, (parallels[1][i,6], parallels[0][i,6]), color=text_color, alpha=text_alpha, size=8)

    for i in range(-90,91,15):
        if i != 0:
            ax.annotate(i, (meridians[1][i + 90,0], meridians[0][i + 90,0]), color=text_color, alpha=text_alpha, size=8)

    for i in range(-75,76,15):
        if i != 0:
            ax.annotate(i, (meridians[1][i + 90, 12], meridians[0][i + 90, 12]), color=text_color, alpha=text_alpha, size=8)

    cax = ax.inset_axes((0.8, 0.985, 0.2, 0.015))
    fig.colorbar(image, cax=cax, orientation='horizontal', label=label)

    ax.axis('off')

    plt.title(title, size=16)
    plt.tight_layout()

    if to_file is not None:
        plt.savefig(to_file)
        plt.ion()
        plt.close(fig)


def show_polar_view(data, pole='north', qr=0.45, label=r'$B_{r}$, G', **kwargs):
    nx, ny = data.shape
    xc, yc = (nx - 1) / 2, (ny - 1) / 2
    Rsun = qr * nx

    if pole == 'north':
        view = View(nx, ny, xc, yc, Rsun, -90, 90, 0) ### North pole view
    else:
        view = View(nx, ny, xc, yc, Rsun, 90, -90, 0) ### South pole view

    return show_data(data, view, label=label, **kwargs)


def show_map(data, figsize=(15,8), label=r'$B_{r}$, G', cmap='hmimag', vmin=-50, vmax=50, sine_lat=False, **kwargs):
    if sine_lat:
        extent = (0, 360, -1, 1)
    else:
        extent = (0, 360, -90, 90)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, aspect='auto', **kwargs)

    ax.set_xticks(np.arange(0,361,30), np.arange(0,361,30))
    if sine_lat:
        ax.set_yticks(np.sin(np.arange(-90, 91, 15) * np.pi / 180), np.arange(-90, 91, 15))
    else:
        ax.set_yticks(np.arange(-90, 91, 15), np.arange(-90, 91, 15))

    ax.set_xlabel('Longitude, degrees')
    ax.set_ylabel('Latitude, degrees')

    cax = ax.inset_axes((0.03, 0.1, 0.1, 0.015))
    fig.colorbar(im, cax=cax, orientation='horizontal', label=label)

    ax.grid(True, ls='--', color='black', alpha=0.2)
    plt.tight_layout()


def plot_flux(*args, sine_lat=False, staggered_lat=False):
    plt.figure(figsize=(10,8))

    for flux in args:
        nx = len(flux)

        if sine_lat:
            spread = 2
        else:
            spread = 180

        if staggered_lat:
            dlat = spread / nx
            lati = np.arange(-spread / 2,spread / 2 + dlat / 2, dlat)
            lat = (lati[1:] + lati[:-1]) / 2
        else:
            dlat = spread / (nx - 1)
            lat = np.arange(-spread / 2,spread / 2 + dlat / 2, dlat)

        if sine_lat:
            lat = np.arcsin(lat.clip(-1,1)) * 180 / np.pi

        plt.plot(lat, flux)

    plt.xlim(-90,90)
    plt.ylim(-10,10)

    plt.grid(True)
    plt.tight_layout()
    plt.show()
