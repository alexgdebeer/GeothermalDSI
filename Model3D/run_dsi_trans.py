from src.dsi import run_dsi

from setup import *

Fs = np.load("data/Fs.npy")
Gs = np.load("data/Gs.npy")

Fs_new = []
Gs_new = []

for i in range(len(Fs.T)):
    ps = data_handler_crse.get_pr_pressures(Fs.T[i])
    if np.min(ps) > P_ATM / 1e6:
        Fs_new.append(Fs.T[i])
        Gs_new.append(Gs.T[i])

Fs_new = np.hstack([F_i[:, np.newaxis] for F_i in Fs_new])
Gs_new = np.hstack([G_i[:, np.newaxis] for G_i in Gs_new])

Fs_trans = []
for F_i in Fs_new.T:
    Fs_trans.append(data_handler_crse.log_transform_pressures(F_i))

Fs_trans = np.hstack([F_i[:, np.newaxis] for F_i in Fs_trans])

m_post, C_post = run_dsi(Gs_new, Fs_trans, y, C_e)
Fs_trans_post = np.random.multivariate_normal(m_post, C_post, size=1000)

Fs_post = []
for F_i in Fs_trans_post:
    Fs_post.append(data_handler_crse.inv_transform_pressures(F_i))

Fs_post = np.hstack([F_i[:, np.newaxis] for F_i in Fs_post])

np.save("data/Fs_post_trans", Fs_post)

# Make transformed Fs
# Run DSI
# Sample from posterior predictive distribution
# Transform back

# 0.01

# m_post, C_post = run_dsi(Gs_new, Fs_new, y, C_e)
# Fs_post = np.random.multivariate_normal(m_post, C_post, size=1000)

# np.save("data/m_post", m_post)
# np.save("data/C_post", C_post)
# np.save("data/Fs_post", Fs_post)
