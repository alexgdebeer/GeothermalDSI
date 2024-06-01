include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")


Ne = 10_000

ωs = rand(pr, Ne)
us = hcat([transform(pr, ω) for ω ∈ eachcol(ωs)]...)

ps = hcat([@time F(u) for u ∈ eachcol(us)]...)

y_obs = model_c.B_obs * ps
y_pred = model_c.B_wells * ps

for i ∈ [10, 100, 1_000, 10_000]

    m_post, C_post = run_dsi(y_obs[:, 1:i], y_pred[:, 1:i], d_obs, C_e)

    m_post = reshape(m_post, 9, :)
    σ_post = reshape(sqrt.(diag(C_post)), 9, :)

    h5write("data/results.h5", "dsi_m_$i", Matrix(m_post))
    h5write("data/results.h5", "dsi_s_$i", Matrix(σ_post))

end

# m_post, C_post = run_dsi(y_obs, y_pred, d_obs, C_e)
# post = MvNormal(m_post, C_post)
# post_samples = rand(post, 1000)

# n_well = 2
# d_obs_well = reshape(d_obs, 9, :)[n_well, :]
# F_well = reshape(model_f.B_wells * F_t, 9, :)[n_well, :]

# p_pri_wells = [hcat(
#     [reshape(p, 9, :)[i, :] for p ∈ eachcol(y_pred)]...
# ) for i in 1:9]

# p_post_wells = [hcat(
#     [reshape(p, 9, :)[i, :] for p ∈ eachcol(post_samples)]...
# ) for i in 1:9]

# p_pri_wells = [vcat(repeat([p0], 1000)', p) for p in p_pri_wells]
# p_post_wells = [vcat(repeat([p0], 1000)', p) for p in p_post_wells]

# p_pri_wells = cat(p_pri_wells...; dims=3)
# p_post_wells = cat(p_post_wells...; dims=3)