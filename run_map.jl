include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")


ω0 = vec(rand(pr, 1))

map, C_post, L_post, n_solves = compute_local_lin(grid_c, model_c, pr, d_obs, ω0, C_e_inv)