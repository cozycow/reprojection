import numpy as np
import glob
from os import path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sunpy.visualization.colormaps as cm
from matplotlib import colormaps
from transforms import ToSpherical, Normalize


def show_data(data, view, title='', label=r'$B_{los}$, G', figsize=(10,10), cmap='hmimag', vmin=-200, vmax=200,
              grid_alpha=0.5, grid_color='black', text_alpha=0.7, text_color='black',
              to_file=None, to_folder=None):

    if to_file is not None or to_folder is not None:
        plt.ioff()

    transform = ~(view.transform() + ToSpherical())

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    if 'xc' in view.__dict__:
        ax.add_patch(Circle((view.yc, view.xc), radius=view.rsun, color='black', ls='-', lw=0.5, alpha=1, fill=False)) # limb

    for phi in range(0,360,15):
        theta = np.arange(-90,91)
        grid = transform(theta, phi * np.ones_like(theta), 1)[:-1]
        if phi == 0:
            ls = '-'
        else:
            ls = '--'
        ax.plot(grid[1], grid[0], color=grid_color, ls=ls, lw=0.5, alpha=grid_alpha)
        ax.annotate(str(phi), (grid[1][90], grid[0][90]), color=text_color, alpha=text_alpha, size=8)

    for theta in range(-90,91,15):
        phi = np.arange(0,361)
        grid = transform(theta * np.ones_like(phi), phi, 1)[:-1]
        ax.plot(grid[1], grid[0], color=grid_color, ls='--', lw=0.5, alpha=grid_alpha)

        if theta != 0:
            ax.annotate(str(theta), (grid[1][0], grid[0][0]), color=text_color, alpha=text_alpha, size=8)

        if theta != 0 and theta != -90 and theta != 90:
            ax.annotate(str(theta), (grid[1][180], grid[0][180]), color=text_color, alpha=text_alpha, size=8)

    cax = ax.inset_axes((0.8, 0.985, 0.2, 0.015))
    fig.colorbar(image, cax=cax, orientation='horizontal', label=label)

    ax.axis('off')

    plt.title(title, size=16)
    plt.tight_layout()

    if to_folder is not None:
        files_in_folder = sorted(glob.glob(path.join(to_folder, '*.png')))
        if len(files_in_folder) == 0:
            last_number = 0
        else:
            last_number = int(files_in_folder[-1].split('/')[-1].split('.')[0])
        to_file = path.join(to_folder, '{:04d}'.format(last_number + 1) + '.png')

    if to_file is not None:
        plt.savefig(to_file)
        plt.ion()
        plt.close(fig)
    else:
        return fig, ax


def show_map(data, map, figsize=(15,8), label=r'$B_{r}$, G', cmap='hmimag', vmin=-50, vmax=50, **kwargs):
    if map.sine:
        extent = (0, 360, -1, 1)
    else:
        extent = (0, 360, -90, 90)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, aspect='auto', **kwargs)

    ax.set_xticks(np.arange(0,361,30), np.arange(0,361,30))
    if map.sine:
        ax.set_yticks(np.sin(np.arange(-90, 91, 15) * np.pi / 180), np.arange(-90, 91, 15))
    else:
        ax.set_yticks(np.arange(-90, 91, 15), np.arange(-90, 91, 15))

    ax.set_xlabel('Longitude, degrees')
    ax.set_ylabel('Latitude, degrees')

    cax = ax.inset_axes((0.03, 0.1, 0.1, 0.015))
    fig.colorbar(im, cax=cax, orientation='horizontal', label=label)

    ax.grid(True, ls='--', color='black', alpha=0.2)
    plt.tight_layout()


