import numpy as np
from view import Map, SunView
from diff_rot import *


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
    if correct_dr:
        Q += [DiffRotate(view.date, car_rot, wsyn=view.wsyn, epoch='20260710T030009', crln0=207.18, **kwargs)]

    view_new.update(inplace=True, **kwargs)
    data_ = view_new.reproject(view.radialize(data), view.update(minmu=minmu), *Q, **kwargs)

    if kind == 'north' or kind == 'south':
        data_ = view_new.deradialize(data_)

    return data_, view_new

