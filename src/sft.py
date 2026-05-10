import numpy as np


def apply_boundary(x, boundary, n=2):
    if boundary == 'mirror':
        xl = x[:n][::-1]
        xr = x[-n:][::-1]
    elif boundary == 'periodic':
        xl = x[-n:]
        xr = x[:n]
    else:
        xl = np.zeros(n)
        xr = np.zeros(n)
    return np.append(xl, np.append(x, xr))


def advect(y, vi, dt, dx=1, xi=None, ai=None, boundary='mirror', hires=True):
    def minmod(a, b):
        return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0)

    def maxmod(a, b):
        return np.where(a * b > 0, np.where(np.abs(a) > np.abs(b), a, b), 0)

    def superbee(a, b):
        return maxmod(minmod(a, 2 * b), minmod(2 * a, b))

    if xi is not None:
        dxi = xi[1:] - xi[:-1]
    else:
        dxi = dx * np.ones_like(y)

    if ai is None:
        ai = np.ones_like(vi)
    a = (ai[:-1] + ai[1:]) / 2

    y_ = apply_boundary(y, boundary, n=2)
    ai_ = apply_boundary(ai, boundary, n=1)
    ql, qr = ai_ * y_[:-1], ai_ * y_[1:]

    Fi = vi * np.where(vi > 0, ql[1:-1], qr[1:-1])
    if hires:
        dxi_ = apply_boundary(dxi, boundary, n=1)
        dx = (dxi_[1:] + dxi_[:-1]) / 2
        dq = qr - ql
        dq_ = superbee(np.where(vi >= 0, dq[:-2], dq[2:]), dq[1:-1])
        Fi += 0.5 * np.abs(vi) * (1 - np.abs(vi * dt / dx)) * dq_

    dF_dx = (Fi[1:] - Fi[:-1]) / dxi
    return y - dF_dx / a * dt


def diffuse(y, d, dt, dx=1, xi=None, ai=None, boundary='mirror'):
    from scipy.linalg import solve_banded
    if xi is not None:
        dxi = xi[1:] - xi[:-1]
    else:
        dxi = dx * np.ones_like(y)

    if ai is None:
        ai = np.ones(len(y) + 1)

    al = ai[:-1]
    ar = ai[1:]
    a = (al + ar) / 2

    dxl = (dxi + np.roll(dxi, 1)) / 2
    dxr = (dxi + np.roll(dxi, -1)) / 2

    ql = d * dt * al / a / dxi / dxl / 2
    qr = d * dt * ar / a / dxi / dxr / 2
    ql[0], qr[-1] = 0, 0

    L = - np.roll(ql, -1)
    U = - np.roll(qr, 1)
    A = 1 + ql + qr
    B = y + qr * (np.roll(y, -1) - y) - ql * (y - np.roll(y, 1))
    return solve_banded((1, 1), [U, A, L], B, True, True)


def diffuse_fft(y, d, dt, dx=1):
    from numpy.fft import fft, ifft, fftfreq
    f = fftfreq(len(y))
    q = d * dt / dx ** 2
    p = q * (1 - np.cos(2 * np.pi * f))
    a = fft(y) * (1 - p) / (1 + p)
    return np.real(ifft(a))


def flow(x, a=0.5, x0=None):
    y = np.sin(2 * x * np.pi / 180) + a * np.sin(4 * x * np.pi / 180) + (2 * a - 1) / 3 * np.sin(6 * x * np.pi / 180)
    if x0 is not None:
        y /= flow(x0, a=a)
    return y


def rotation(x, a=14.712, b=-2.396, c=-1.787):
    sin2 = np.sin(x * np.pi / 180) ** 2
    return a + b * sin2 + c * sin2 ** 2

