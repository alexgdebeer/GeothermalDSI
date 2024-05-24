include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")

ω0 = zeros(pr.Nω)

map, C_post, L_post, n_solves = compute_local_lin(grid_c, model_c, pr, d_obs, ω0, C_e_inv)

ω_post = map.ω .+ L_post * rand(Normal(), (50, 1000))
u_post = hcat([transform(pr, ω) for ω ∈ eachcol(ω_post)]...)
p_post = hcat([F(u) for u ∈ eachcol(u_post)]...)

p_post_wells = [hcat(
    [reshape(p, 9, :)[i, :] for p ∈ eachcol(model_c.B_wells * p_post)]...
) for i in 1:9]

p_post_wells = [vcat(repeat([p0], 1000), p) for p in p_post_wells]
p_post_wells = cat(p_post_wells...; dims=3)