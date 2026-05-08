import numpy as np


def advect(y, vi, dt, dx=1, xi=None, ai=None, boundary='reflect'):
    def superbee(x):
        return np.max([(2 * x).clip(0,1), x.clip(0,2)], axis=0)

    if xi is not None:
        dxi = xi[1:] - xi[:-1]
        dx = (np.roll(xi, -1) - np.roll(xi, 1)) / 2
    else:
        dxi = dx

    if ai is None:
        ai = a = 1
    else:
        a = (ai[:-1] + ai[1:]) / 2

    if boundary == 'reflect':
        yl = yr = 0.
    elif boundary == 'periodic':
        yl, yr = y[[-1,0]]
    elif boundary == 'constant':
        yl, yr = y[[0,-1]]
    else:
        yl, yr = boundary

    ql, qr = ai * np.append(yl, y), ai * np.append(y, yr)
    dq = qr - ql

    ri = np.where(vi > 0, np.roll(dq, 1) * dq / (dq ** 2 + 1e-16), np.roll(dq, -1) * dq / (dq ** 2 + 1e-16))
    Fi = vi * np.where(vi > 0, ql, qr) + 0.5 * np.abs(vi) * (1 - np.abs(vi * dt / dx)) * superbee(ri) * dq

    dF_dx = (Fi[1:] - Fi[:-1]) / dxi
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
    qr = d * dt * ar / a / dxr / dxi / 2 * np.ones_like(y)
    ql[0] = 0
    qr[-1] = 0

    L = - np.roll(ql, -1)
    U = - np.roll(qr, 1)
    A = 1 + ql + qr
    B = y + qr * (np.roll(y, -1) - y) - ql * (y - np.roll(y, 1))
    return solve_banded((1, 1), [U, A, L], B, True, True)


def diffuse_fft(y, d, dt, dx=1):
    from numpy.fft import fft, ifft, fftfreq
    q = d * dt / dx ** 2
    f = fftfreq(len(y))
    a = fft(y) / (1 + 2 * q * (1 - np.cos(2 * np.pi * f)))
    return np.real(ifft(a))


def flow(x, a=0.5, x0=None):
    y = np.sin(2 * x * np.pi / 180) + a * np.sin(4 * x * np.pi / 180) + (2 * a - 1) / 3 * np.sin(6 * x * np.pi / 180)
    if x0 is not None:
        y /= flow(x0, a=a)
    return y


def rotation(x, a=14.712, b=-2.396, c=-1.787):
    sin2 = np.sin(x * np.pi / 180) ** 2
    return a + b * sin2 + c * sin2 ** 2

