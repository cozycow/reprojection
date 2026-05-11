import numpy as np


def advect(y, vi, dx, dt, ai=None):
    if ai is None:
        ai = np.ones(y.shape + (1,))
    a = (ai[1:] + ai[:-1]) / 2

    Fi = vi * ai * np.where(vi > 0, np.append(0, y), np.append(y, 0))
    return y - (Fi[1:] - Fi[:-1]) / a * dt / dx


def diffuse(y, d, dx, dt, ai=None):
    from scipy.linalg import solve_banded
    if ai is None:
        ai = np.ones(y.shape + (1,))

    a = (ai[1:] + ai[:-1]) / 2

    L = - d * ai[1:] * dt / dx ** 2
    L[-1] = 0

    U = - d * ai[:-1] * dt / dx ** 2
    U[0] = 0

    A = a - np.roll(U, -1) - np.roll(L, 1)
    B = a * y

    return solve_banded((1, 1), [U, A, L], B, True, True)
