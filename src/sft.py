import numpy as np


def advect(y, vi, dt, dx=1, xi=None, ai=None, boundary='mirror'):
    if xi is not None:
        dx = xi[1:] - xi[:-1]

    if ai is None:
        ai = a = 1
    else:
        a = (ai[:-1] + ai[1:]) / 2

    if boundary == 'mirror':
        yl = yr = 0.
    else:
        yl, yr = y[[-1,0]]

    Fi = vi * ai * np.where(vi > 0, np.append(yl, y), np.append(y, yr))
    dF_dx = (Fi[1:] - Fi[:-1]) / dx
    return y - dF_dx / a * dt


def diffuse(y, d, dt, dx=1, xi=None, ai=None):
    from scipy.linalg import solve_banded
    if xi is not None:
        x = (xi[1:] + xi[:-1]) / 2
        dxl = x - np.roll(x, 1)
        dxr = np.roll(x, -1) - x
        dxi = xi[1:] - xi[:-1]
    else:
        dxi = dxl = dxr = dx

    if ai is None:
        a = al = ar = 1
    else:
        al = ai[:-1]
        ar = ai[1:]
        a = (al + ar) / 2

    ql = d * dt * al / a / dxl / dxi / 2 * np.ones_like(y)
    ql[0] = 0

    qr = d * dt * ar / a / dxr / dxi / 2 * np.ones_like(y)
    qr[-1] = 0

    L = - np.roll(ql, -1)
    U = - np.roll(qr, 1)
    A = 1 + ql + qr
    B = y + qr * (np.roll(y, -1) - y) - ql * (y - np.roll(y, 1))
    return solve_banded((1, 1), [U, A, L], B, True, True)


def flow(x, a=0.5, x0=None):
    y = np.sin(2 * x * np.pi / 180) + a * np.sin(4 * x * np.pi / 180) + (2 * a - 1) / 3 * np.sin(6 * x * np.pi / 180)
    if x0 is not None:
        y /= flow(x0, a=a)
    return y
