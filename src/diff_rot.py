import numpy as np
from transforms import Rotate

ALPHA = 1.327124e20
WSID = 360 / 25.38
A, B, C = 14.712, -2.396, -1.787 # differential rotation rates (Snodgrass & Ulrich, ApJ, 351, 309, 1990)


def diff_rot2(r, *, dsun_obs, obs_vn, obs_vw, obs_vr, crln_obs, crlt_obs):

    theta_obs, a, e = kepler(dsun_obs, np.sqrt(obs_vw ** 2 + obs_vn ** 2), obs_vr)
    E_obs = ecc_anomaly(theta_obs, e)
    M_obs = mean_anomaly(E_obs, e)
    w_obs = np.sqrt(ALPHA / a ** 3) * 24 * 3600

    rot = ~Rotate.x(crlt_obs) + Rotate.y(crln_obs)
    r_obs = rot(0, 0, 1)
    crln_obs = np.arctan2(r_obs[0], r_obs[2]) * 180 / np.pi
    axis_obs = rot(-obs_vn, obs_vw, 0)
    r_ = Rotate.from_axis_angle(axis_obs, -theta_obs)(*r_obs)
    delta = M_obs + np.arctan2(r_[0], r_[2]) * 180 / np.pi
    delta = (delta - crln_obs + 180) % 360 - 180

    #delta = 0

    crln = np.arctan2(r[0], r[2]) * 180 / np.pi
    time = ((crln - crln_obs + 180) % 360 - 180 - delta) / (WSID - w_obs)

    print(w_obs, 360 / (WSID - w_obs))
    #time = ((crln - delta + 180) % 360 - 180) / (WSID - w_obs)
    time = np.nan_to_num(time)

    time[np.abs(time) > 360 / (WSID - w_obs) / 4] = np.nan

    w = diff_rot(r)

    return w[0] * time, w[1] * time, w[2] * time


def diff_rot(r, *, a=A-WSID, b=B, c=C):
    x, y, z = r
    return 0, a + b * y ** 2 + c * y ** 4, 0


def kepler(r, vt, vr):
    p = (r * vt) ** 2 / ALPHA
    v0 = np.sqrt(ALPHA / p)
    e = np.sqrt((vt / v0 - 1) ** 2 + (vr / v0) ** 2)
    a = p / (1 - e ** 2)

    theta = np.arctan2(vr, vt - v0) * 180 / np.pi % 360
    return theta, a, e


def ecc_anomaly(theta, e):
    cos_f = np.cos(theta * np.pi / 180)
    sin_f = np.sin(theta * np.pi / 180)
    return np.arctan2(np.sqrt(1 - e ** 2) * sin_f, e + cos_f) * 180 / np.pi


def mean_anomaly(E, e):
    return E - e * np.sin(E * np.pi / 180) * 180 / np.pi

