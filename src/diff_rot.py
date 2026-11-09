import numpy as np

ALPHA = 1.327124e20
WSID = 360 / 25.38
A, B, C = 14.712, -2.396, -1.787 # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)


def diff_rot2(r, *, delta, wsyn):
    crln = np.arctan2(r[0], r[2]) * 180 / np.pi

    print(WSID - wsyn, 360 / wsyn)
    time = ((crln - delta + 180) % 360 - 180) / wsyn
    time = np.nan_to_num(time)

    w = diff_rot(r)
    return w[0] * time, w[1] * time, w[2] * time


def diff_rot(r, *, a=A-WSID, b=B, c=C):
    x, y, z = r
    return 0, a + b * y ** 2 + c * y ** 4, 0

