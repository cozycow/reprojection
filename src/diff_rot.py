import numpy as np
from transforms import *

WSID = 360 / 25.38
A, B, C = 14.712, -2.396, -1.787 # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)
def diff_rot(crlt, *, a=A, b=B, c=C):
    y = np.sin(crlt * np.pi / 180)
    return a + b * y ** 2 + c * y ** 4


class DiffRotate(Transform):

    def __init__(self, date, crn=None,
                 wsyn=360 / 27.2753,
                 epoch='18531109T000000',
                 crln0=0.):

        self.date = date
        self.crn = crn
        self.wsyn = wsyn
        self.epoch = epoch
        self.crln0 = crln0

    def __repr__(self):
        return f'CorrectDR()'

    def __call__(self, theta, phi, alpha):
        from datetime import datetime
        cdn = (datetime.fromisoformat(self.date) -
              datetime.fromisoformat(self.epoch)).total_seconds() / 24 / 3600

        phi0 = cdn * self.wsyn + 360 - self.crln0
        delta = (phi + phi0 + 180) % 360 - 180

        if self.crn is not None:
            crn = (phi0 - delta) / 360

            t = np.where(np.any([crn < self.crn,
                                 crn >= self.crn + 1], axis=0))

            #t = np.where(crn != self.crn)
            theta[t], phi[t], alpha[t] = np.nan, np.nan, np.nan

        time = delta / self.wsyn
        w = diff_rot(theta) - WSID
        dphi = w * time

        return theta, phi + dphi, alpha

    def __invert__(self):
        return self

