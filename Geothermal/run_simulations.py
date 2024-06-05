import numpy as np
from scipy import sparse

from src.consts import SECS_PER_WEEK
from src.data_handlers import DataHandler
from src.grfs import *
from src.models import *
from src.priors import Prior

# np.random.seed(1)


MODEL_FOLDER = "models"
DATA_FOLDER = "data"

TRUTH_FOLDER = f"{DATA_FOLDER}/truth"
GRF_FOLDER_2D = f"{DATA_FOLDER}/grf_2d"
GRF_FOLDER_3D = f"{DATA_FOLDER}/grf_3d"

W_TRUE_PATH = f"{TRUTH_FOLDER}/w_true.npy"
P_TRUE_PATH = f"{TRUTH_FOLDER}/p_true.npy"
F_TRUE_PATH = f"{TRUTH_FOLDER}/F_true.npy"
G_TRUE_PATH = f"{TRUTH_FOLDER}/G_true.npy"
OBS_PATH = f"{TRUTH_FOLDER}/obs.npy"
COV_PATH = f"{TRUTH_FOLDER}/C_e.npy"

READ_TRUTH = False

MESH_PATH_CRSE = f"{MODEL_FOLDER}/gFL8788"
MESH_PATH_FINE = f"{MODEL_FOLDER}/gFL13383"

MODEL_PATH_CRSE = f"{MODEL_FOLDER}/FL8788"
MODEL_PATH_FINE = f"{MODEL_FOLDER}/FL13383"

GRF_FOLDER_2D_CRSE = f"{GRF_FOLDER_2D}/FL8788"
GRF_FOLDER_2D_FINE = f"{GRF_FOLDER_2D}/FL13383"

GRF_FOLDER_3D_CRSE = f"{GRF_FOLDER_3D}/FL8788"
GRF_FOLDER_3D_FINE = f"{GRF_FOLDER_3D}/FL13383"

"""
Model parameters
"""

tmax, nt = 104.0 * SECS_PER_WEEK, 24
dt = tmax / nt

mesh_crse = Mesh(MESH_PATH_CRSE)
mesh_fine = Mesh(MESH_PATH_FINE)

well_xs = [1800, 3000, 4000, 2800, 2100, 3800,  3300, 1400, 4600]
well_ys = [3100, 3000,  3000, 1900, 3900, 2100, 4100, 1500, 4600]
n_wells = len(well_xs)
well_depths = [-2600] * n_wells
feedzone_depths = [-1200] * n_wells
feedzone_rates = [[[0.0, -0.25], [tmax/2, -0.5]]] * n_wells

wells_crse = [Well(x, y, depth, mesh_crse, fz_depth, fz_rate)
              for (x, y, depth, fz_depth, fz_rate) 
              in zip(well_xs, well_ys, well_depths, feedzone_depths, feedzone_rates)]

wells_fine = [Well(x, y, depth, mesh_fine, fz_depth, fz_rate)
              for (x, y, depth, fz_depth, fz_rate) 
              in zip(well_xs, well_ys, well_depths, feedzone_depths, feedzone_rates)]

"""
Clay cap
"""

# Bounds for depth of clay cap, width (horizontal and vertical) and dip
bounds_geom_cap = [(-900, -775), (1400, 1600), (200, 300), (300, 600)]
n_terms = 5
coef_sds = 5

clay_cap_crse = ClayCap(mesh_crse, bounds_geom_cap, n_terms, coef_sds)
clay_cap_fine = ClayCap(mesh_fine, bounds_geom_cap, n_terms, coef_sds)

"""
Permeability fields
"""

def levels_ext(p):
    """Level set mapping for all non-fault and non-cap regions."""
    if   p < -1.5: return -15.5
    elif p < -0.5: return -15.0
    elif p <  0.5: return -14.5
    elif p <  1.5: return -14.0
    else: return -13.5

def levels_flt(p):
    """Level set mapping for fault."""
    if   p < -0.5: return -13.5
    elif p <  0.5: return -13.0
    else: return -12.5

def levels_cap(p):
    """Level set mapping for clay cap."""
    if   p < -0.5: return -17.0
    elif p <  0.5: return -16.5
    else: return -16.0

std_ext = 1.25
std_flt = 0.75
std_cap = 0.75
l_perm = np.array([8000, 8000, 1000])

kernel_perm = SquaredExp(l_perm)

grf_perm_crse = GRF3D(mesh_crse, kernel_perm, folder=GRF_FOLDER_3D_CRSE)
grf_perm_fine = GRF3D(mesh_fine, kernel_perm, folder=GRF_FOLDER_3D_FINE)

perm_field_ext_crse = PermField(std_ext, grf_perm_crse, levels_ext)
perm_field_flt_crse = PermField(std_flt, grf_perm_crse, levels_flt)
perm_field_cap_crse = PermField(std_cap, grf_perm_crse, levels_cap)

perm_field_ext_fine = PermField(std_ext, grf_perm_fine, levels_ext)
perm_field_flt_fine = PermField(std_flt, grf_perm_fine, levels_flt)
perm_field_cap_fine = PermField(std_cap, grf_perm_fine, levels_cap)

"""
Fault
"""

mu_upflow = 2.5e-4
std_upflow = 0.5e-4
l_upflow = 500

kernel_upflow = SquaredExp(l_upflow)

grf_upflow_crse = GRF2D(mesh_crse, kernel_upflow, folder=GRF_FOLDER_2D_CRSE)
grf_upflow_fine = GRF2D(mesh_fine, kernel_upflow, folder=GRF_FOLDER_2D_FINE)

bounds_fault = [(1500, 4500), (1500, 4500)]

fault_crse = Fault(mesh_crse, bounds_fault)
fault_fine = Fault(mesh_fine, bounds_fault)

upflow_field_crse = UpflowField(mu_upflow, std_upflow, grf_upflow_crse)
upflow_field_fine = UpflowField(mu_upflow, std_upflow, grf_upflow_fine)

# Parameter associated with Gaussian kernel for upflows
ls_upflows = 1600

"""
Observations
"""

temp_obs_zs = [-800, -1100, -1400, -1700, -2000, -2300, -2600]
temp_obs_cs = np.array([[x, y, z] 
                        for z in temp_obs_zs 
                        for x, y in zip(well_xs, well_ys)])

prod_obs_ts = np.array([0, 13, 26, 39, 52]) * SECS_PER_WEEK

data_handler_crse = DataHandler(mesh_crse, wells_crse, temp_obs_cs, prod_obs_ts, tmax, nt)
data_handler_fine = DataHandler(mesh_fine, wells_fine, temp_obs_cs, prod_obs_ts, tmax, nt)

"""
Ensemble functions
"""

def generate_particle(p_i, num):
    name = f"{MODEL_PATH_CRSE}_{num}"
    logks_t, upflows_t = prior.split(p_i)
    model = Model(name, mesh_crse, logks_t, wells_crse, upflows_t, dt, tmax)
    return model

def get_result(particle: Model):
    F_i = particle.get_pr_data()
    G_i = data_handler_crse.get_obs(F_i)
    return F_i, G_i

"""
Prior
"""

prior = Prior(
    mesh_crse, clay_cap_crse, fault_crse, 
    perm_field_ext_crse, perm_field_flt_crse, perm_field_cap_crse, 
    upflow_field_crse, ls_upflows)

"""
Truth generation
"""

noise_level = 0.05

truth_dist = Prior(
    mesh_fine, clay_cap_fine, fault_fine, 
    perm_field_ext_fine, perm_field_flt_fine, perm_field_cap_fine, 
    upflow_field_fine, ls_upflows)

def generate_truth():

    p_t = truth_dist.sample()
    logks_t, upflows_t = truth_dist.split(p_t)

    model_t = Model(
        MODEL_PATH_FINE, mesh_fine, 
        logks_t, wells_fine, upflows_t, dt, tmax
    )

    grf_perm_fine.plot(logks_t)
    grf_perm_fine.slice_plot(logks_t)
    
    if model_t.run() != ExitFlag.SUCCESS:
        raise Exception("Truth failed to run.")

    F_t = model_t.get_pr_data()
    G_t = data_handler_fine.get_obs(F_t)

    np.save(P_TRUE_PATH, p_t)
    np.save(F_TRUE_PATH, F_t)
    np.save(G_TRUE_PATH, G_t)

    return w_t, p_t, F_t, G_t

def read_truth():

    p_t = np.load(P_TRUE_PATH)
    F_t = np.load(F_TRUE_PATH)
    G_t = np.load(G_TRUE_PATH)
    
    return w_t, p_t, F_t, G_t

def generate_data(G_t):

    temp_t, pres_t, enth_t = data_handler_fine.split_obs(G_t)

    cov_temp = (noise_level * np.max(temp_t)) ** 2 * np.eye(temp_t.size)
    cov_pres = (noise_level * np.max(pres_t)) ** 2 * np.eye(pres_t.size)
    cov_enth = (noise_level * np.max(enth_t)) ** 2 * np.eye(enth_t.size)

    C_e = sparse.block_diag((cov_temp, cov_pres, cov_enth)).toarray()
    y = np.random.multivariate_normal(G_t, C_e)

    np.save(OBS_PATH, y)
    np.save(COV_PATH, C_e)

    return y, C_e

def read_data():
    y = np.load(OBS_PATH)
    C_e = np.load(COV_PATH)
    return y, C_e

if READ_TRUTH:
    w_t, p_t, F_t, G_t = read_truth()
    y, C_e = read_data()

else:
    w_t, p_t, F_t, G_t = generate_truth()
    y, C_e = generate_data(G_t)

Np = mesh_crse.m.num_cells + mesh_crse.m.num_columns
NF = mesh_crse.m.num_cells + 2 * n_wells * (nt+1)
NG = len(y)