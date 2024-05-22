include("setup.jl")


function run_dsi(
    Xs::AbstractMatrix, 
    Ys::AbstractMatrix,
    x_obs::AbstractVector
)

    Nx = size(Xs, 1)

    m_x = mean(Xs, dims=2)
    m_y = mean(Ys, dims=2)

    C_joint = cov(vcat(Xs, Ys)')
    C_xx = C_joint[1:Nx, 1:Nx]
    C_yy = C_joint[Nx+1:end, Nx+1:end]
    C_yx = C_joint[Nx+1:end, 1:Nx]

    m_post = m_y + C_yx * (C_xx \ (x_obs - m_x))
    C_post = C_yy - C_yx * (C_xx \ C_yx')

    return vec(m_post), Hermitian(C_post) + 1e-4 * Diagonal(diag(C_post))

end

Ne = 1000

θs = rand(pr, Ne)
us = hcat([transform(pr, θ) for θ ∈ eachcol(θs)]...)

ps = hcat([F(u) for u ∈ eachcol(us)]...)
es = rand(e_dist, Ne)

y_obs = model_c.B_obs * ps + es
y_pred = model_c.B_preds * ps

m_post, C_post = run_dsi(y_obs, y_pred, d_obs)
post = MvNormal(m_post, C_post)
post_samples = rand(post, 1000)

n_well = 2
d_obs_well = reshape(d_obs, 9, :)[n_well, :]
F_well = reshape(model_f.B_wells * F_t, 9, :)[n_well, :]

pri_samples_well = hcat(
    [reshape(s, 9, :)[n_well, :] for s ∈ eachcol(y_pred)]...
)

post_samples_well = hcat(
    [reshape(s, 9, :)[n_well, :] for s ∈ eachcol(post_samples)]...
)

plot(t_preds, post_samples_well)
plot!(t_preds, pri_samples_well)
plot!(2:2:160, F_well)
scatter!(t_obs, d_obs_well)