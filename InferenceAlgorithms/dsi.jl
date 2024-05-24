function run_dsi(
    Xs::AbstractMatrix, 
    Ys::AbstractMatrix,
    x_obs::AbstractVector;
    jitter::Bool=true
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
    C_post = Hermitian(C_post)

    if jitter
        C_post += 1e-8 * Diagonal(diag(C_post))
    end

    return vec(m_post), C_post

end