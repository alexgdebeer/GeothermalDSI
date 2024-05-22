using LinearAlgebra
using LinearSolve
using HDF5
using SparseArrays


function SciMLBase.solve(
    g::Grid, 
    m::Model,
    u::AbstractVector
)::AbstractVector

    p = zeros(g.nx^2, g.nt)

    Au = m.ϕ * m.c * sparse(I, g.nx^2, g.nx^2) + 
        (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u)) * g.∇h
    
    b = g.Δt * m.Q[:, 1] .+ m.ϕ * m.c * m.p0
    p[:, 1] = solve(LinearProblem(Au, b))

    for t ∈ 2:g.nt
        b = g.Δt * m.Q[:, t] + m.ϕ * m.c * p[:, t-1]
        p[:, t] = solve(LinearProblem(Au, b))
    end

    return vec(p)

end


function SciMLBase.solve(
    g::Grid, 
    m::AbstractModel,
    u::AbstractVector,
    np_r::Int,
    μ_pi::AbstractVector,
    V_ri::AbstractMatrix
)::AbstractVector

    p_r = zeros(np_r, g.nt)

    Au = m.ϕ * m.c * sparse(I, g.nx^2, g.nx^2) + 
        (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u)) * g.∇h
    Au_r = V_ri' * Au * V_ri

    b = V_ri' * (g.Δt * m.Q[:, 1] .+ (m.ϕ * m.c * m.p0) .- Au * μ_pi)
    p_r[:, 1] = solve(LinearProblem(Au_r, b))

    for t ∈ 2:g.nt
        bt = V_ri' * (g.Δt * m.Q[:, t] - Au * μ_pi) + m.ϕ * m.c * (p_r[:, t-1] + V_ri' * μ_pi)
        p_r[:, t] = solve(LinearProblem(Au_r, bt))
    end

    return vec(V_ri * p_r .+ μ_pi)

end


SciMLBase.solve(g::Grid, m::ReducedOrderModel, u::AbstractVector) = 
    SciMLBase.solve(g, m, u, m.np_r, m.μ_pi, m.V_ri)

function generate_pod_samples(
    g::Grid,
    m::Model,
    pr::MaternField,
    N::Int
)::AbstractMatrix

    θs = rand(pr, N)
    us = [transform(pr, θ_i) for θ_i ∈ eachcol(θs)]
    ps = hcat([@time solve(g, m, u_i) for u_i ∈ us]...)
    return ps

end


function compute_pod_basis(
    g::Grid,
    ps::AbstractMatrix,
    var_to_retain::Real
)::Tuple{AbstractVector, AbstractMatrix}

    ps_reshaped = reshape(ps, g.nx^2, :)'

    μ = vec(mean(ps_reshaped, dims=1))
    Γ = cov(ps_reshaped)

    eigendecomp = eigen(Γ, sortby=(λ -> -λ))
    Λ, V = eigendecomp.values, eigendecomp.vectors

    N_r = findfirst(cumsum(Λ)/sum(Λ) .> var_to_retain)
    V_r = V[:, 1:N_r]
    @info "Reduced basis computed (dimension: $N_r)."

    return μ, V_r

end


function generate_pod_data(
    g::Grid,
    m::Model,
    pr::MaternField,
    N::Int, 
    var_to_retain::Real,
    fname::AbstractString
)

    ps_samp = generate_pod_samples(g, m, pr, N)
    μ_pi, V_ri = compute_pod_basis(g, ps_samp, var_to_retain)

    h5write("data/$fname.h5", "μ_pi", μ_pi)
    h5write("data/$fname.h5", "V_ri", V_ri)

    return μ_pi, V_ri
    
end


function read_pod_data(
    fname::AbstractString
)

    f = h5open("data/$fname.h5")

    μ_pi = f["μ_pi"][:]
    V_ri = f["V_ri"][:, :]

    return μ_pi, V_ri

end