import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from reprojection import *


def show_data(data, view, title='', label='', figsize=(10,10), cmap='seismic', vmin=-200, vmax=200, to_file=None):
    transform = ~(view.to_carrington() + ToSpherical())

    grid = np.mgrid[-90:90.5:1,0:360.5:1]
    grid, alpha = transform(grid)
    grid = np.array(grid)

    meridians = grid[:,:,::15]
    parallels = np.transpose(grid[:,::15,:], (0,2,1))

    if to_file is not None:
        plt.ioff()

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(meridians[1][:,1:], meridians[0][:,1:], color='black', ls='--', lw=0.5, alpha=0.5)
    ax.plot(meridians[1][:,0], meridians[0][:,0], color='black', ls='-', lw=0.5, alpha=1) # central meridian

    ax.plot(parallels[1][:,:6], parallels[0][:,:6], color='black', ls='--', lw=0.5, alpha=0.5)
    ax.plot(parallels[1][:,6], parallels[0][:,6], color='black', ls='-', lw=0.5, alpha=1) # equator
    ax.plot(parallels[1][:,7:], parallels[0][:,7:], color='black', ls='--', lw=0.5, alpha=0.5)

    ax.add_patch(Circle((view.yc, view.xc), radius=view.rsun, color='black', ls='-', lw=0.5, alpha=1, fill=False))

    for i in range(0,360,15):
        ax.annotate(i, (parallels[1][i,6], parallels[0][i,6]), alpha=0.7, size=8)

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
        plt.close(fig)
