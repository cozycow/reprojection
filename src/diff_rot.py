import numpy as np
from transforms import *
from datetime import datetime


WSID = 360 / 25.38
A, B, C = 14.712, -2.396, -1.787 # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)
def diff_rot(x, *, a=A, b=B, c=C, **kwargs):
    y = np.sin(x * np.pi / 180)
    return a + b * y ** 2 + c * y ** 4


class DiffRotate(Transform):

    def __init__(self, date,
                 wsyn=360 / 27.2753,
                 epoch='18531109T000000',
                 crln0=0.,
                 **kwargs):

        self.date = date
        self.wsyn = wsyn
        self.epoch = epoch
        self.crln0 = crln0
        self.kwargs = kwargs

    def __repr__(self):
        return f'DiffRotate()'

    def __call__(self, theta, phi):
        cdn = (datetime.fromisoformat(self.date) -
              datetime.fromisoformat(self.epoch)).total_seconds() / 24 / 3600

        phi0 = cdn * self.wsyn + 360 - self.crln0
        delta = (phi + phi0 + 180) % 360 - 180
        time = delta / self.wsyn
        w = diff_rot(theta, **self.kwargs) - WSID
        dphi = w * time
        return theta, phi + dphi

    def __invert__(self):
        return self


class FilterRotation(Transform):

    def __init__(self, date, crn,
                 wsyn=360 / 27.2753,
                 epoch='18531109T000000',
                 crln0=0.,
                 **kwargs):

        self.date = date
        self.crn = crn
        self.wsyn = wsyn
        self.epoch = epoch
        self.crln0 = crln0

    def __repr__(self):
        return f'FilterRotation()'

    def __call__(self, theta, phi):
        cdn = (datetime.fromisoformat(self.date) -
              datetime.fromisoformat(self.epoch)).total_seconds() / 24 / 3600

        phi0 = cdn * self.wsyn + 360 - self.crln0
        delta = (phi + phi0 + 180) % 360 - 180
        crn = (phi0 - delta) / 360
        t = np.where(np.any([crn < self.crn,
                             crn >= self.crn + 1], axis=0))
        theta[t], phi[t] = np.nan, np.nan
        return theta, phi

    def __invert__(self):
        return self


