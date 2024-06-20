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

# np.save("data/Fs_new.npy", Fs_new)

n_samples = [10, 100, 250, 500, 600, 676]

for n in n_samples:
    
    m_post, C_post = run_dsi(Gs_new[:, :n], Fs_new[:, :n], y, C_e)
    Fs_post = np.random.multivariate_normal(m_post, C_post, size=1000)

    np.save(f"data/m_post_{n}", m_post)
    np.save(f"data/C_post_{n}", C_post)
    np.save(f"data/Fs_post_{n}", Fs_post)