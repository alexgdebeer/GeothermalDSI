using LinearAlgebra
using LinearSolve
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