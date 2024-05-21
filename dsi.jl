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
    C_post = C_yy - C_yx * (C_xx \ C_xy)

    return m_post, C_post

end



Ne = 1000

θs = rand(pr, Ne)
us = hcat([transform(pr, θ) for θ ∈ eachcol(θs)]...)

ps = hcat([F(u) for u ∈ eachcol(us)]...)
ds = model_r.B * ps # TODO: add some noise to these.
qs = model_r.P * ps

m_post, C_post = run_dsi(ds, qs)