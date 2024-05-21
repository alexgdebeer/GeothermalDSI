using Distributions
using LinearAlgebra


struct MaternField

    m::AbstractVector
    C::AbstractMatrix 
    L::AbstractMatrix

    Nθ::Int
    
    function MaternField(
        g::Grid,
        m::Real,
        σ::Real,
        l::Real
    )
        
        m = fill(m, g.nx^2)
        C = zeros(g.nx^2, g.nx^2)

        cs = [[xi, xj] for xi ∈ g.xs, xj ∈ g.xs]
        for (i, ci) ∈ enumerate(cs), (j, cj) ∈ enumerate(cs)
            dx = norm(ci-cj)
            C[i, j] = σ^2 * (1 + √3*dx/l) * exp(-√3*dx/l)
        end

        L = cholesky(C + 1e-8I).L

        return new(m, C, L, g.nx^2)

    end

end


function Base.rand(
    mf::MaternField, 
    n::Int=1
)
    return rand(Normal(), mf.Nθ, n)
end


function transform(
    mf::MaternField,
    θ::AbstractVecOrMat
)
    return mf.m + mf.L * θ
end