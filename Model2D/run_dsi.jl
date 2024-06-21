include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")


Ne = 1_000

ωs = rand(pr, Ne)
us = hcat([transform(pr, ω) for ω ∈ eachcol(ωs)]...)

ps = hcat([@time F(u) for u ∈ eachcol(us)]...)

y_obs = model_c.B_obs * ps
y_pred = model_c.B_wells * ps

m_post, C_post = run_dsi(y_obs, y_pred, d_obs, C_e)

post = MvNormal(m_post, C_post)
post_samples = rand(post, 1000)

p_post_wells = [hcat(
    [reshape(p, 9, :)[i, :] for p ∈ eachcol(post_samples)]...
) for i in 1:9]

p_post_wells = [vcat(fill(p0, 1000)', p) for p in p_post_wells]
p_post_wells = cat(p_post_wells...; dims=3)

h5write("data/results.h5", "dsi_preds", p_post_wells)