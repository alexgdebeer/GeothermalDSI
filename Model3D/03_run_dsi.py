import numpy as np

from src.dsi import DSIMapping

from setup import *


Fs = np.load("data/Fs.npy")
Gs = np.load("data/Gs.npy")

es = np.random.multivariate_normal(
    mean=np.zeros(len(y)), 
    cov=C_e, 
    size=Gs.shape[1]
).T
Gs += es
GFs = np.vstack([Gs, Fs])

RUN_DSI = True
RUN_SAMPLE_COMPARISON = True
RUN_VALIDATION = True


if RUN_DSI:

    dsi_mapping = DSIMapping(GFs[:, :1300])
    Fs_pri = dsi_mapping.generate_prior_samples(ny=len(y), n=1000)
    Fs_post = dsi_mapping.generate_conditional_samples(y, n=1000)

    np.save(f"data/Fs_pri", Fs_pri)
    np.save(f"data/Fs_post", Fs_post)


if RUN_SAMPLE_COMPARISON:

    n_samples = [100, 250, 500, 1000, 1300]

    for n in n_samples:
        
        dsi_mapping = DSIMapping(GFs[:, :n])
        Fs_post = dsi_mapping.generate_conditional_samples(y, n=1000)
        
        np.save(f"data/Fs_post_{n}", Fs_post)


if RUN_VALIDATION: 
    
    dsi_mapping = DSIMapping(GFs[:, :1300])

    Fs_v = Fs[:, 1300:1400]
    Gs_v = Gs[:, 1300:1400]

    in_bounds = np.zeros((Fs_v.shape[0], 100), dtype=bool)

    for i, (G_i, F_i) in enumerate(zip(Gs_v.T, Fs_v.T)):
        
        Fs_post_v = dsi_mapping.generate_conditional_samples(G_i, n=1000)
        
        quantiles = np.quantile(Fs_post_v, q=(0.025, 0.975), axis=1)

        in_bounds[:, i] = np.bitwise_and(quantiles[0] < F_i, F_i < quantiles[1])

    in_bounds_temp = in_bounds[data_handler_crse.ind_pr_temp]
    in_bounds_pres = in_bounds[data_handler_crse.inds_pr_pres, 13:]
    in_bounds_enth = in_bounds[data_handler_crse.inds_pr_enth, 13:]

    print(np.mean(in_bounds_temp))
    print(np.mean(in_bounds_pres))
    print(np.mean(in_bounds_enth))