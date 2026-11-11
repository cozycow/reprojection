import numpy as np
import glob
from os import path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sunpy.visualization.colormaps as cm
from matplotlib import colormaps
from transforms import ToSpherical
from view import *


def plot(x, y, ax, lim=50, label=None, **kwargs):
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    dl = np.sqrt(dx ** 2 + dy ** 2)
    t = np.where(dl > lim)[0]

    if len(t) > 0:
        plot(x[:t[0] + 1], y[:t[0] + 1], ax, **kwargs)
        plot(x[t[0] + 1:], y[t[0] + 1:], ax, **kwargs)

        ax.annotate(label, (x[t[0]], y[t[0]]), color='black', alpha=1, size=8, ha='center', va='center')
        ax.annotate(label, (x[t[0] + 1], y[t[0] + 1]), color='black', alpha=1, size=8, ha='center', va='center')
    else:
        ax.plot(x, y, **kwargs)


def draw_colorbar(fig, ax, im, label=None, loc='upper right'):
    cbbox = inset_axes(ax, width=1.5, height=0.6, loc=loc)

    [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
    cbbox.tick_params(
        axis='both',
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False
    )

    cbbox.set_facecolor((1, 1, 1, 0.7))
    cax = cbbox.inset_axes((0.05, 0.7, 0.9, 0.15))
    cax.tick_params(labelsize=8)
    fig.colorbar(im, cax=cax, orientation='horizontal', label=label)


def show_data(data, view, title='', label=r'$B_{los}$, G', dpi=120, cmap='hmimag', vmin=-200, vmax=200,
              grid_alpha=0.5, grid_color='black', text_alpha=0.7, text_color='black', cbloc='upper right',
              to_file=None, to_folder=None):

    if to_file is not None or to_folder is not None:
        plt.ioff()

    transform = ~(view.transform() + ToSpherical())

    fig, ax = plt.subplots(figsize=(view.ny / dpi, view.nx / dpi))
    ax.axis('off')
    ax.set_title(title, size=16)
    image = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    draw_colorbar(fig, ax, image, label=label, loc=cbloc)

    if isinstance(view, View):
        ax.add_patch(Circle((view.yc, view.xc), radius=view.rsun, color='black', lw=0.5, fill=False)) # limb
    else:
        ax.add_patch(Rectangle((0,0), width=view.ny-1, height=view.nx-1, color='black', lw=0.5, fill=False))


    for phi in range(0,360,15):
        theta = np.arange(-90,91)
        grid = transform(theta, phi * np.ones_like(theta), 1)
        plot(grid[1], grid[0], ax, color=grid_color, ls='--', lw=0.5, alpha=grid_alpha)

        if isinstance(view, View):
            theta0 = 0.
        elif isinstance(view, Map):
            if view.cassini:
                theta0 = 0.
            else:
                theta0 = -89.9
        else:
            theta0 = np.nan
        grid = transform(theta0, phi, 1)

        ax.annotate(str(phi) if phi != 0 else None, (grid[1], grid[0]), color=text_color, alpha=text_alpha,
                    size=8, ha='right', va='top')

    for theta in range(-90,91,15):
        phi = np.arange(0,360)
        grid = transform(theta * np.ones_like(phi), phi, 1)
        plot(grid[1], grid[0], ax, color=grid_color, ls='--', lw=0.5, alpha=grid_alpha,
             label=str(theta) if isinstance(view, View) and theta != 0 else None)

        ax.annotate(str(theta), (grid[1][0], grid[0][0]), color=text_color, alpha=text_alpha,
                    size=8, ha='right', va='top')

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

