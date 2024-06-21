from src.dsi import run_dsi

from setup import *

RUN_DSI = True 
RUN_SAMPLE_COMPARISON = True


Fs = np.load("data/Fs.npy")
Gs = np.load("data/Gs.npy")


if RUN_DSI:

    m_post, C_post = run_dsi(Gs, Fs, y, C_e)
    Fs_post = np.random.multivariate_normal(m_post, C_post, size=1000)

    np.save(f"data/m_post", m_post)
    np.save(f"data/C_post", C_post)
    np.save(f"data/Fs_post", Fs_post)


if RUN_SAMPLE_COMPARISON:

    n_samples = [10, 100, 250, 500, 600, 676]

    for n in n_samples:
        
        m_post, C_post = run_dsi(Gs[:, :n], Fs[:, :n], y, C_e)

        np.save(f"data/m_post_{n}", m_post)
        np.save(f"data/C_post_{n}", C_post)