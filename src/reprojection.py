import numpy as np
from view import ViewBase, SunView
from diff_rot import *


class Map(ViewBase):

    def __init__(self, nx, ny, sine=True, cassini=False):
        self.nx = nx
        self.ny = ny
        self.sine = sine
        self.cassini = cassini

    @property
    def transform(self):
        transform = Normalize(((self.nx - 1) / 2, 0), ((self.nx - 1) / 180, self.ny / 360))

        if self.sine:
            transform -= Sine()

        if self.cassini:
            transform += ~ToSpherical() + SwapAxes.xy() + ToSpherical()

        return transform

    @property
    def grid(self):
        grid = tuple(np.mgrid[:self.nx, :self.ny].astype(np.float32))
        return grid


class View(SunView):

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

    @classmethod
    def load_preset(cls, preset='north'):
        if preset == 'north':
            return cls(nx=1024, ny=1024, xc=511.5, yc=511.5, rsun=480, rsun_deg=0,
                       crota=-90, crlt=90, crln=0, minmu=-1e-8)
        elif preset == 'south':
            return cls(nx=1024, ny=1024, xc=511.5, yc=511.5, rsun=480, rsun_deg=0,
                       crota=90, crlt=-90, crln=0, minmu=-1e-8)
        elif preset == 'warp':
            return cls(nx=1024, ny=1024, xc=511.5, yc=511.5, rsun=480, rsun_deg=-45,
                       crota=0, crlt=0, crln=0, minmu=-1e-8)


def reproject(data, header,
              header_new=None,
              kind='',
              minmu=0.1,
              car_rot=None,
              correct_dr=False,
              full_output=False,
              deradialize=False,
              **kwargs):

    view = View.from_header(header)

    if header_new is not None:
        view_new = View.from_header(header_new)
    elif kind == 'synoptic':
        nx, ny = 720, 1800
        view_new = Map(nx, ny, sine=True)
    elif kind in ['north', 'south', 'warp']:
        view_new = View.load_preset(kind)
    else:
        view_new = view.copy()

    Q = []

    if car_rot is not None:
        Q += [FilterRotation(view.date, car_rot, **kwargs)]

    if correct_dr:
        Q += [DiffRotate(view.date, **kwargs)]

    view_new.update(inplace=True, **kwargs)
    data_ = view_new.reproject(view.radialize(data), view.update(minmu=minmu), *Q, full_output=full_output)

    if deradialize:
        if full_output:
            data_[0] = view_new.deradialize(data_[0])
        else:
            data_ = view_new.deradialize(data_)

    if full_output:
        mu = view.get_mu(data_[1], data_[2])
        data_ = np.concat([data_, np.expand_dims(mu, 0)], axis=0)

    return data_, view_new

