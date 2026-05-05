import numpy as np


def advect(y, xi, vi, dt, ai=None):
    if ai is None:
        ai = np.ones_like(xi)
    a = (ai[1:] + ai[:-1]) / 2

    Fi = vi * ai * np.where(vi > 0, np.append(0, y), np.append(y, 0))
    dy = (Fi[1:] - Fi[:-1]) / (xi[1:] - xi[:-1])
    return y - dy / a * dt


def diffuse(y, xi, d, dt, ai=None):
    from scipy.linalg import solve_banded
    if ai is None:
        ai = np.ones_like(xi)

    a = (ai[1:] + ai[:-1]) / 2
    x = (xi[1:] + xi[:-1]) / 2

    dx = x - np.roll(x, 1)
    dxi = xi[1:] - xi[:-1]

    L = - d * dt * ai[1:] / np.roll(dx, -1) / np.roll(dxi, -1)
    L[-1] = 0
    U = - d * dt * ai[:-1] / dx / np.roll(dxi, 1)
    U[0] = 0

    A = a - np.roll(U, -1) - np.roll(L, 1)
    B = a * y
    return solve_banded((1, 1), [U, A, L], B, True, True)


def flow(x, a=0.5, x0=None):
    y = np.sin(2 * x * np.pi / 180) + a * np.sin(4 * x * np.pi / 180) + (2 * a - 1) / 3 * np.sin(6 * x * np.pi / 180)
    if x0 is not None:
        y /= flow(x0, a=a)
    return y
