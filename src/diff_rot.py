import numpy as np


WSID = 360 / 25.38
A, B, C = 14.712, -2.396, -1.787 # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)


def diff_rot2(r, *, delta, wsyn, wsid=WSID):
    crln = np.arctan2(r[0], r[2]) * 180 / np.pi
    time = ((crln - delta + 180) % 360 - 180) / wsyn
    time = np.nan_to_num(time)
    w = diff_rot(r, a=A, b=B, c=C, wsid=wsid)
    return w * time


def diff_rot(r, *, a=A, b=B, c=C, wsid=WSID):
    x, y, z = r
    return a + b * y ** 2 + c * y ** 4 - wsid

