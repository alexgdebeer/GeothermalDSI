from src.dsi import run_dsi

from setup import *

Fs = np.load("data/Fs.npy")
Gs = np.load("data/Gs.npy")

print(Fs.shape)

Fs_new = []
Gs_new = []

for i in range(len(Fs.T)):
    ps = data_handler_crse.get_full_pressures(Fs.T[i])
    if np.min(ps) > P_ATM / 1e6:
        Fs_new.append(Fs.T[i])
        Gs_new.append(Gs.T[i])

Fs_new = np.hstack([F_i[:, np.newaxis] for F_i in Fs_new])
Gs_new = np.hstack([G_i[:, np.newaxis] for G_i in Gs_new])

print(Fs_new.shape)

m_post, C_post = run_dsi(Gs_new, Fs_new, y, C_e)
Fs_post = np.random.multivariate_normal(m_post, C_post, size=1000)

np.save("data/m_post", m_post)
np.save("data/C_post", C_post)
np.save("data/Fs_post", Fs_post)

# from matplotlib import pyplot as plt

# for F_i in Fs_post:
#     pressures = data_handler_crse.get_full_pressures(F_i)
#     plt.plot(pressures.T[0])

# plt.show()

# print(m_post)

# print(Fs.shape)
# print(Gs.shape)

# print(NF)
# print(NG)