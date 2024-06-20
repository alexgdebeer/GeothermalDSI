struct MaternField

    μ::AbstractVector
    C::AbstractMatrix 
    λs::AbstractVector 
    vs::AbstractMatrix

    Nθ::Int
    Nω::Int
    
    function MaternField(
        g::Grid,
        m::Real,
        σ::Real,
        l::Real,
        Nω::Int=50
    )
        
        μ = fill(m, g.nx^2)
        C = zeros(g.nx^2, g.nx^2)

        cs = [[xi, xj] for xi ∈ g.xs, xj ∈ g.xs]
        for (i, ci) ∈ enumerate(cs), (j, cj) ∈ enumerate(cs)
            C[i, j] = σ^2 * exp(-0.5 * (ci-cj)' * (ci-cj) / l^2)
        end

        C += 1e-8I

        # Compute eigenvalues and eigenvectors
        eigendecomp = eigen(C, sortby=(λ)->(-λ))
        λs = eigendecomp.values[1:Nω]
        vs = eigendecomp.vectors[:, 1:Nω]

        return new(μ, C, λs, vs, g.nx^2, Nω)

    end

end


function Base.rand(
    mf::MaternField, 
    n::Int=1
)
    return rand(Normal(), mf.Nω, n)
end


function transform(
    mf::MaternField,
    ωs::AbstractVecOrMat
)
    return mf.μ + mf.vs * (sqrt.(mf.λs) .* ωs) 
end