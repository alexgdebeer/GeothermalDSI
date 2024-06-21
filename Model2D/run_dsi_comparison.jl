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