include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

NF = grid_c.nx^2 * grid_c.nt
Ni = 500_000

β = 0.05

Nb = 100
Nc = 5

run_pcn(
    F, G, pr, d_obs, L_e, 
    NF, Ni, Nb, Nc, β
)