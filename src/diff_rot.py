import numpy as np
from transforms import Rotate


WSID = 360 / 25.38
A, B, C = 14.712, -2.396, -1.787 # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)


def diff_rot2(r, *, dsun_obs, obs_vn, obs_vw, obs_vr, crln_obs, crlt_obs, a=A-WSID, b=B, c=C):
    crln = np.arctan2(r[0], r[2]) * 180 / np.pi

    time = get_time(crln, dsun_obs, obs_vn, obs_vw, obs_vr, crln_obs, crlt_obs)
    time = -np.nan_to_num(time)
    w = diff_rot(r, a=a, b=b, c=c)

    return w[0] * time, w[1] * time, w[2] * time


def diff_rot(r, *, a=A-WSID, b=B, c=C):
    x, y, z = r
    return 0, a + b * y ** 2 + c * y ** 4, 0


def ecc_anomaly(theta, e):
    cos_f = np.cos(theta * np.pi / 180)
    sin_f = np.sin(theta * np.pi / 180)
    return np.arctan2(np.sqrt(1 - e ** 2) * sin_f, e + cos_f) * 180 / np.pi % 360 + theta // 360 * 360


def orbital_time(theta, a, e, alpha=1.327124e20):
    E = ecc_anomaly(theta, e) * np.pi / 180
    return a * np.sqrt(a / alpha) * (E - e * np.sin(E))


def kepler(r, vt, vr, alpha=1.327124e20):
    p = (r * vt) ** 2 / alpha
    v0 = np.sqrt(alpha / p)
    e = np.sqrt((vt / v0 - 1) ** 2 + (vr / v0) ** 2)
    a = p / (1 - e ** 2)

    theta = np.arctan2(vr, vt - v0) * 180 / np.pi % 360
    return theta, a, e


def get_time(crln, dsun_obs, obs_vn, obs_vw, obs_vr, crln_obs, crlt_obs):
    theta_obs, a, e = kepler(dsun_obs, np.sqrt(obs_vw ** 2 + obs_vn ** 2), obs_vr)
    t_obs = orbital_time(theta_obs + 360, a, e)

    phi = np.arange(-180, 180, 0.1)
    theta = theta_obs + phi

    t = orbital_time(theta + 360, a, e)
    dt = (t - t_obs) / 24 / 3600

    rot = ~Rotate.x(crlt_obs) + Rotate.y(crln_obs)
    r_obs = rot((0, 0, 1))[0]
    axis_obs = rot((-obs_vn, obs_vw, 0))[0]

    r_ = Rotate.from_axis_angle(axis_obs, phi)(r_obs)[0]
    delta_ = (np.arctan2(r_[0], r_[2]) * 180 / np.pi - dt * WSID) % 360

    for k in np.where(delta_[1:] > delta_[:-1])[0]:
        delta_[k+1:] -= 360
    delta_ -= delta_[len(delta_) // 2]

    delta = (crln - crln_obs + 180) % 360 - 180
    return np.interp(-delta, -delta_, dt)