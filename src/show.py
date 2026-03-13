import numpy as np
import matplotlib.pyplot as plt


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