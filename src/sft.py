import numpy as np


def advect(y, xi, vi, dt, ai=None, boundary='mirror'):
    if ai is None:
        ai = np.ones_like(xi)
    a = (ai[1:] + ai[:-1]) / 2

    if boundary == 'mirror':
        yl = yr = 0.
    else:
        yl, yr = y[[-1,0]]

    Fi = vi * ai * np.where(vi > 0, np.append(yl, y), np.append(y, yr))
    dF_dx = (Fi[1:] - Fi[:-1]) / (xi[1:] - xi[:-1])
    return y - dF_dx / a * dt


def diffuse(y, xi, d, dt, ai=None):
    from scipy.linalg import solve_banded
    if ai is None:
        ai = np.ones_like(xi)

    a = (ai[1:] + ai[:-1]) / 2
    x = (xi[1:] + xi[:-1]) / 2

    dx = x - np.roll(x, 1)
    dxi = xi[1:] - xi[:-1]

    ql = d * dt * ai[:-1] / a / dx / dxi / 2
    ql[0] = 0

    qr = d * dt * ai[1:] / a / np.roll(dx, -1) / dxi / 2
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
