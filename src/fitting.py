import numpy as np


def polyfit2d(f, x=None, y=None, x_out=None, y_out=None, degree=1, weights=None, terms=None, sigma=None, niter=1, return_coefficients=False):
    def terms_(x, y, degree=1):
        if degree == 1:
            return np.array([x, y])
        return np.append(terms_(x, y, degree=degree-1), np.array([x ** (degree - i) * y ** i for i in range(degree + 1)]), axis=0)

    if x is None and y is None:
        nx, ny = f.shape
        x, y = np.mgrid[-nx // 2:nx // 2, -ny // 2:ny // 2].astype(np.float32)
        x /= nx / 2
        y /= ny / 2

    if x_out is None and y_out is None:
        x_out = x
        y_out = y

    if weights is None:
        weights = np.ones_like(f)

    if terms is None:
        terms = lambda x, y: terms_(x, y, degree=degree)

    t = np.where(~np.isnan(f))

    X = terms(x[t], y[t])
    Y = f[t]
    W = weights[t]

    k, b = 0, 0

    for _ in range(niter):
        X0 = np.mean(X * W, axis=-1, keepdims=True) / np.mean(W)
        Y0 = np.mean(Y * W) / np.mean(W)

        X_ = X - X0
        Y_ = Y - Y0

        k = (Y_ * W) @ X_.T @ np.linalg.inv((X_ * W) @ X_.T)
        b = Y0 - k @ X0

        if sigma is not None:
            W = 1 / np.abs(Y - np.sum([k_ * x_ for k_, x_ in zip(k, X)]) - b).clip(sigma)


    if return_coefficients:
        return k, b
    else:
        return np.sum([k_ * x_ for k_, x_ in zip(k, terms(x_out, y_out))], axis=0) + b