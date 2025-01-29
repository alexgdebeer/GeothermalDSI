import numpy as np

from src.dsi import DSIMapping

import os
# os.chdir("Model3D")

print("Setting up...")
from setup import *
print("Finished setup.")


RUN_DSI = True
RUN_SAMPLE_COMPARISON = True
RUN_VALIDATION = True


Fs = np.load("data/Fs.npy")
Gs = np.load("data/Gs.npy")

es = np.random.multivariate_normal(
    mean=np.zeros(len(y)), 
    cov=C_e, 
    size=Gs.shape[1]
).T
Gs += es
Gs = np.maximum(Gs, 0.0) 

GFs = np.vstack([Gs, Fs])

t_min, t_max = -20.0, 380.0
p_min, p_max = -1.0, 15.0
e_min, e_max = -100, 3500.0

Gs_min = np.zeros(Gs.shape[0])
Gs_max = np.zeros(Gs.shape[0])

Fs_min = np.zeros(Fs.shape[0])
Fs_max = np.zeros(Fs.shape[0])

Gs_min[data_handler_crse.inds_obs_temp] = t_min 
Gs_max[data_handler_crse.inds_obs_temp] = t_max 
Gs_min[data_handler_crse.inds_obs_pres] = p_min 
Gs_max[data_handler_crse.inds_obs_pres] = p_max 
Gs_min[data_handler_crse.inds_obs_enth] = e_min 
Gs_max[data_handler_crse.inds_obs_enth] = e_max

Fs_min[data_handler_crse.inds_ns_temp] = t_min
Fs_min[data_handler_crse.inds_pr_temp] = t_min
Fs_max[data_handler_crse.inds_ns_temp] = t_max
Fs_max[data_handler_crse.inds_pr_temp] = t_max
Fs_min[data_handler_crse.inds_pr_pres] = p_min 
Fs_max[data_handler_crse.inds_pr_pres] = p_max 
Fs_min[data_handler_crse.inds_pr_enth] = e_min 
Fs_max[data_handler_crse.inds_pr_enth] = e_max

GFs_min = np.concatenate((Gs_min, Fs_min))
GFs_max = np.concatenate((Gs_max, Fs_max))


if RUN_DSI:

    dsi_mapping = DSIMapping(GFs[:, :1300], GFs_min, GFs_max)
    Fs_pri = dsi_mapping.generate_prior_samples(ny=len(y), n=1000)
    Fs_post = dsi_mapping.generate_conditional_samples(y, n=1000)

    np.save(f"data/Fs_pri", Fs_pri)
    np.save(f"data/Fs_post", Fs_post)


if RUN_SAMPLE_COMPARISON:

    n_samples = [100, 250, 500, 1000, 1300]

    for n in n_samples:
        
        dsi_mapping = DSIMapping(GFs[:, :n], GFs_min, GFs_max)
        Fs_post = dsi_mapping.generate_conditional_samples(y, n=1000)
        
        np.save(f"data/Fs_post_{n}", Fs_post)


if RUN_VALIDATION: 
    
    dsi_mapping = DSIMapping(GFs[:, :1300], GFs_min, GFs_max)

    Fs_v = Fs[:, 1300:1400]
    Gs_v = Gs[:, 1300:1400]

    in_bounds = np.zeros((Fs_v.shape[0], 100), dtype=bool)

    for i, (G_i, F_i) in enumerate(zip(Gs_v.T, Fs_v.T)):

        print(i)
        
        Fs_post_v = dsi_mapping.generate_conditional_samples(G_i, n=1000)
        
        quantiles = np.quantile(Fs_post_v, q=(0.025, 0.975), axis=1)

        in_bounds[:, i] = np.bitwise_and(quantiles[0] < F_i, F_i < quantiles[1])

    in_bounds_temp = in_bounds[data_handler_crse.ind_pr_temp]
    in_bounds_pres = in_bounds[data_handler_crse.inds_pr_pres, 13:]
    in_bounds_enth = in_bounds[data_handler_crse.inds_pr_enth, 13:]

    print(np.mean(in_bounds_temp))
    print(np.mean(in_bounds_pres))
    print(np.mean(in_bounds_enth))