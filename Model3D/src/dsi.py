import numpy as np


def run_dsi(Xs, Ys, x_obs, C_e, jitter=False):

    Nx = Xs.shape[0]

    m_x = np.mean(Xs, axis=1)
    m_y = np.mean(Ys, axis=1)

    C_joint = np.cov(np.vstack([Xs, Ys]))
    C_xx = C_joint[:Nx, :Nx]
    C_yy = C_joint[Nx:, Nx:]
    C_yx = C_joint[Nx:, :Nx]

    m_post = m_y + C_yx @ np.linalg.solve(C_xx + C_e, x_obs - m_x)
    C_post = C_yy - C_yx @ np.linalg.solve(C_xx + C_e, C_yx.T)

    if jitter:
        C_post += 1e-8 * np.diag(np.diag(C_post))

    return m_post, C_post