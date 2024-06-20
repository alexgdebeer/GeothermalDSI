from src.dsi import run_dsi

from setup import *


Fs = np.load("data/Fs.npy")
Gs = np.load("data/Gs.npy")

Fs_trans = []
for F_i in Fs.T:
    Fs_trans.append(data_handler_crse.log_transform_pressures(F_i))

Fs_trans = np.hstack([F_i[:, np.newaxis] for F_i in Fs_trans])

m_post, C_post = run_dsi(Gs, Fs_trans, y, C_e)
Fs_post_trans = np.random.multivariate_normal(m_post, C_post, size=1000)

Fs_post = []
for F_i in Fs_post_trans:
    Fs_post.append(data_handler_crse.inv_transform_pressures(F_i))

Fs_post = np.hstack([F_i[:, np.newaxis] for F_i in Fs_post])

np.save("data/Fs_post_trans", Fs_post)