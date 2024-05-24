include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")


Ne = 1000

θs = rand(pr, Ne)
us = hcat([transform(pr, θ) for θ ∈ eachcol(θs)]...)

ps = hcat([F(u) for u ∈ eachcol(us)]...)
es = rand(e_dist, Ne)

y_obs = model_c.B_obs * ps + es
y_pred = model_c.B_wells * ps

m_post, C_post = run_dsi(y_obs, y_pred, d_obs)
post = MvNormal(m_post, C_post)
post_samples = rand(post, 1000)

n_well = 2
d_obs_well = reshape(d_obs, 9, :)[n_well, :]
F_well = reshape(model_f.B_wells * F_t, 9, :)[n_well, :]

p_pri_wells = [hcat(
    [reshape(p, 9, :)[i, :] for p ∈ eachcol(y_pred)]...
) for i in 1:9]

p_post_wells = [hcat(
    [reshape(p, 9, :)[i, :] for p ∈ eachcol(post_samples)]...
) for i in 1:9]

p_pri_wells = [vcat(repeat([p0], 1000)', p) for p in p_pri_wells]
p_post_wells = [vcat(repeat([p0], 1000)', p) for p in p_post_wells]

p_pri_wells = cat(p_pri_wells...; dims=3)
p_post_wells = cat(p_post_wells...; dims=3)