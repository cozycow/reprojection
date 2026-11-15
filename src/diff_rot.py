import numpy as np


WSID = 360 / 25.38
WSYN = 360 / 27.2753
A, B, C = 14.712, -2.396, -1.787 # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)

def get_crn(crln, date='18531109T000000'):
    from datetime import datetime
    dt = (datetime.fromisoformat(date) -
    datetime(1853, 11, 9, 0, 0, 0))

    cdn = dt.total_seconds() / 24 / 3600
    crn = cdn / 27.2753 + 1

    delta = crn + crln / 360
    delta = (delta + 0.5) % 1 - 0.5
    #delta -= np.round(delta)
    return crn - delta


def diff_rot3(crlt, crln, *, date):
    from datetime import datetime
    dt = (datetime.fromisoformat(date) -
          datetime(1853, 11, 9, 0, 0, 0))

    cdn = dt.total_seconds() / 24 / 3600
    crln0 = -cdn * WSYN

    delta = crln - crln0
    delta = (delta + 180) % 360 - 180

    time = delta / WSYN
    time = np.nan_to_num(time)
    w = diff_rot(crlt, a=A, b=B, c=C) - WSID
    return w * time


def diff_rot2(y, crln, *, delta, date, wsyn):
    time = (crln + get_crn(delta, date) * 360) / wsyn
    time = np.nan_to_num(time)
    w = diff_rot(y, a=A, b=B, c=C) - WSID
    return w * time


def diff_rot(crlt, *, a=A, b=B, c=C):
    y = np.sin(crlt * np.pi / 180)
    return a + b * y ** 2 + c * y ** 4

