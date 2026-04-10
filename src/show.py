import numpy as np
import matplotlib.pyplot as plt
from reprojection import *


def show_polar_data(data, figsize=(10,10), label='', **kwargs):
    fig = plt.figure(figsize=figsize)

    axes_coords = (0.05, 0.05, 0.9, 0.9)

    ax_image = fig.add_axes(axes_coords)
    im = ax_image.imshow(data, origin='lower', **kwargs)

    ax_image.axis('off')

    cax = ax_image.inset_axes((0.8, 0.985, 0.2, 0.015))
    fig.colorbar(im, cax=cax, orientation='horizontal', label=label)

    ax_polar = fig.add_axes(axes_coords, projection = 'polar')
    ax_polar.patch.set_alpha(0)

    lats = np.arange(15,91,15)

    ax_polar.set_rorigin(0)
    ax_polar.set_rticks(np.cos(lats * np.pi / 180))
    ax_polar.set_rlabel_position(-90)
    ax_polar.set_yticklabels([rf'{lat}$\degree$' for lat in lats], color='black')

    ax_polar.grid(True, ls='--', color='black', alpha=0.6)

    return fig, ax_image


def show_data(data, view, title='', label='', figsize=(10,10), cmap='seismic', vmin=-200, vmax=200, to_file=None):
    transform = ~(view.to_carrington() + ToSpherical())

    grid = np.mgrid[-90:90.5:1,-180:180.5:1]
    grid, alpha = transform(grid)
    grid = np.array(grid)

    meridians = grid[:,:,::15]
    parallels = np.transpose(grid[:,::15,:], (0,2,1))

    if to_file is not None:
        plt.ioff()

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(meridians[1][:,:-1], meridians[0][:,:-1], color='black', ls='--', lw=0.5, alpha=0.5)
    ax.plot(parallels[1][:,:6], parallels[0][:,:6], color='black', ls='--', lw=0.5, alpha=0.5)
    ax.plot(parallels[1][:,6], parallels[0][:,6], color='black', ls='-', lw=0.5, alpha=1) # equator
    ax.plot(parallels[1][:,7:], parallels[0][:,7:], color='black', ls='--', lw=0.5, alpha=0.5)

    for i in range(-180,180,15):
        ax.annotate(i, (parallels[1][i + 180,6], parallels[0][i + 180,6]), alpha=0.7, size=8)

    for i in range(-90,91,15):
        if i != 0:
            ax.annotate(i, (meridians[1][i + 90,0], meridians[0][i + 90,0]), alpha=0.7, size=8)
            ax.annotate(i, (meridians[1][i + 90, 12], meridians[0][i + 90, 12]), alpha=0.7, size=8)

    cax = ax.inset_axes((0.8, 0.985, 0.2, 0.015))
    fig.colorbar(image, cax=cax, orientation='horizontal', label=label)

    ax.axis('off')

    plt.title(title, size=16)
    plt.tight_layout()

    if to_file is not None:
        plt.savefig(to_file)
        plt.ion()
