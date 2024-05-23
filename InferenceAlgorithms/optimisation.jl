const GN_MIN_NORM = 1e-2
const GN_MAX_ITS = 30
const CG_MAX_ITS = 30
const LS_C = 1e-4
const LS_MAX_ITS = 20

const UNIT_NORM = Normal()


struct GNResult

    converged::Bool
    θ::AbstractVector
    u::AbstractVector
    p::AbstractVector

    Au::AbstractMatrix
    ∂Ap∂u::AbstractMatrix

    n_solves::Int

end

function J(
    θ::AbstractVector, 
    p::AbstractVector,
    y::AbstractVector,
    m::Model,
    C_e_inv::AbstractMatrix
)

    res = m.B_obs * p - y
    return 0.5 * res' * C_e_inv * res + 0.5 * sum(θ.^2)

end

function compute_∂Ap∂u(
    u::AbstractVector, 
    p::AbstractVector,
    g::Grid,
    m::Model
)

    p = reshape(p, g.nx^2, g.nt)

    ∂Au∂u = (g.Δt / m.μ) * vcat([
        g.∇h' * spdiagm(g.∇h * p[:, t]) * 
        g.A * spdiagm(exp.(u)) for t ∈ 1:g.nt
    ]...)

    return ∂Au∂u

end

function solve_forward(
    Au::AbstractMatrix,
    g::Grid,
    m::Model
)

    p = zeros(g.nx^2, g.nt)
    
    b = g.Δt * m.Q[:, 1] .+ m.ϕ * m.c * m.p0
    p[:, 1] = solve(LinearProblem(Au, b))

    for t ∈ 2:g.nt
        b = g.Δt * m.Q[:, t] + m.ϕ * m.c * p[:, t-1]
        p[:, t] = solve(LinearProblem(Au, b))
    end

    return vec(p)

end

function solve_adjoint(
    p::AbstractVector, 
    Au::AbstractMatrix,
    y::AbstractVector,
    g::Grid,
    m::Model,
    C_e_inv::AbstractMatrix
)

    λ = zeros(g.nx^2, g.nt)

    b = -m.B_obs' * C_e_inv * (m.B_obs * p - y)
    b = reshape(b, g.nx^2, g.nt)

    λ[:, end] = solve(LinearProblem(Matrix(Au'), b[:, end])).u

    for t ∈ (g.nt-1):-1:1
        bt = b[:, t] + m.c * m.ϕ * λ[:, t+1]
        λ[:, t] = solve(LinearProblem(Matrix(Au'), bt)).u
    end

    return vec(λ)

end

function compute_∂Ax∂θtx(
    ∂Ax∂u::AbstractMatrix, 
    x::AbstractVector,
    pr::MaternField
)

    return (∂Ax∂u * pr.vs * spdiagm(sqrt.(pr.λs)))' * x

end

function compute_∂Ax∂θx(
    ∂Ax∂u::AbstractMatrix, 
    x::AbstractVector,
    pr::MaternField
)

    return ∂Ax∂u * pr.vs * spdiagm(sqrt.(pr.λs)) * x

end

function compute_∇Lθ(
    ∂Ap∂u::AbstractMatrix,
    θ::AbstractVector,
    λ::AbstractVector,
    pr::MaternField
)

    return θ + compute_∂Ax∂θtx(∂Ap∂u, λ, pr)

end

function solve_forward_inc(
    Au::AbstractMatrix,
    b::AbstractVector,
    g::Grid,
    m::Model
)

    b = reshape(b, g.nx^2, g.nt)
    p = zeros(g.nx^2, g.nt)

    p[:, 1] = solve(LinearProblem(Au, b[:, 1])).u

    for t ∈ 2:g.nt
        bt = b[:, t] + m.ϕ * m.c * p[:, t-1]
        p[:, t] = solve(LinearProblem(Au, bt)).u
    end

    return vec(p)

end

function solve_adjoint_inc(
    Au::AbstractMatrix,
    b::AbstractVector,
    g::Grid,
    m::Model
)
    
    Au_t = Matrix(Au')
    b = reshape(b, g.nx^2, g.nt)

    λ = zeros(g.nx^2, g.nt)
    λ[:, end] = solve(LinearProblem(Au_t, b[:, end])).u

    for t ∈ (g.nt-1):-1:1
        bt = b[:, t] + m.c * m.ϕ * λ[:, t+1]
        λ[:, t] = solve(LinearProblem(Au_t, bt)).u
    end

    return vec(λ)

end

function compute_Hmx(
    x::AbstractVector,
    Au::AbstractMatrix, 
    ∂Ap∂u::AbstractMatrix,
    g::Grid,
    m::Model,
    pr::MaternField,
    C_e_inv::AbstractMatrix
)

    ∂Ap∂θx = compute_∂Ax∂θx(∂Ap∂u, x, pr)

    p_inc = solve_forward_inc(Au, ∂Ap∂θx, g, m)
    b_inc = m.B_obs' * C_e_inv * m.B_obs * p_inc
    λ_inc = solve_adjoint_inc(Au, b_inc, g, m)

    ∂Ap∂θtλ_inc = compute_∂Ax∂θtx(∂Ap∂u, λ_inc, pr)
    return ∂Ap∂θtλ_inc

end

function compute_Hx(
    x::AbstractVector,
    Au::AbstractMatrix, 
    ∂Ap∂u::AbstractMatrix,
    g::Grid,
    m::Model,
    pr::MaternField,
    C_e_inv::AbstractMatrix
)

    return compute_Hmx(x, Au, ∂Ap∂u, g, m, pr, C_e_inv) + x

end

function solve_cg(
    Au::AbstractMatrix,
    ∂Ap∂u::AbstractMatrix,
    ∇Lθ::AbstractVector,
    tol::Real,
    g::Grid,
    m::Model,
    pr::MaternField,
    C_e_inv::AbstractMatrix
)

    println("CG It. | norm(r)")
    δθ = spzeros(pr.Nω)
    d = -copy(∇Lθ)
    r = -copy(∇Lθ)

    i = 0
    while true

        i += 1

        Hd = compute_Hx(d, Au, ∂Ap∂u, g, m, pr, C_e_inv)

        α = (r' * r) / (d' * Hd)
        δθ += α * d

        r_prev = copy(r)
        r = r_prev - α * Hd

        @printf "%6i | %.3e\n" i norm(r)

        if norm(r) < tol
            println("CG converged after $i iterations.")
            return δθ, i
        elseif i > CG_MAX_ITS
            @warn "CG failed to converge within $CG_MAX_ITS iterations."
            return δθ, i
        end
        
        β = (r' * r) / (r_prev' * r_prev)
        d = r + β * d

    end

end

function linesearch(
    θ_c::AbstractVector, 
    p_c::AbstractVector, 
    ∇Lθ_c::AbstractVector, 
    δθ::AbstractVector,
    y::AbstractVector,
    g::Grid, 
    m::Model,
    pr::MaternField,
    C_e_inv
)

    println("LS It. | J(η, u)")
    J_c = J(θ_c, p_c, y, m, C_e_inv)
    
    α_k = 1.0
    θ_k = nothing
    p_k = nothing

    i = 0
    while i < LS_MAX_ITS

        i += 1

        θ_k = θ_c + α_k * δθ
        u_k = transform(pr, θ_k)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u_k)) * g.∇h

        p_k = solve_forward(Au, g, m)
        J_k = J(θ_k, p_k, y, m, C_e_inv)

        @printf "%6i | %.3e\n" i J_k 

        if (J_k ≤ J_c + LS_C * α_k * ∇Lθ_c' * δθ)
            println("Linesearch converged after $i iterations.")
            return θ_k, p_k, i
        end

        α_k *= 0.5

    end

    @warn "Linesearch failed to converge within $LS_MAX_ITS iterations."
    return θ_k, p_k, i

end

function compute_map(
    g::Grid,
    m::Model,
    pr::MaternField,
    y::AbstractVector,
    θ0::AbstractVector,
    C_e_inv::AbstractMatrix
)

    θ = copy(θ0) 
    p = nothing
    norm∇Lθ0 = nothing
    
    i = 0
    n_solves = 0
    while true

        i += 1

        @info "Beginning GN It. $i"
        
        u = transform(pr, θ)

        Au = m.c * m.ϕ * sparse(I, g.nx^2, g.nx^2) + 
            (g.Δt / m.μ) * g.∇h' * spdiagm(g.A * exp.(u)) * g.∇h

        if i == 1
            p = solve_forward(Au, g, m)
            n_solves += 1
        end
        λ = solve_adjoint(p, Au, y, g, m, C_e_inv)
        n_solves += 1

        ∂Ap∂u = compute_∂Ap∂u(u, p, g, m)

        ∇Lθ = compute_∇Lθ(∂Ap∂u, θ, λ, pr)
        if i == 1
            norm∇Lθ0 = norm(∇Lθ)
        end
        tol_cg = min(0.5, √(norm(∇Lθ) / norm∇Lθ0)) * norm(∇Lθ)

        if norm(∇Lθ) < 1e-5 * norm∇Lθ0
            @info "Converged."
            return GNResult(true, θ, u, p, Au, ∂Ap∂u, n_solves)
        end

        @printf "norm(∇Lθ): %.3e\n" norm(∇Lθ)
        @printf "CG tolerance: %.3e\n" tol_cg

        δθ, n_it_cg = solve_cg(Au, ∂Ap∂u, ∇Lθ, tol_cg, g, m, pr, C_e_inv)
        θ, p, n_it_ls = linesearch(θ, p, ∇Lθ, δθ, y, g, m, pr, C_e_inv)
        
        n_solves += (2n_it_cg + n_it_ls)

        if i > GN_MAX_ITS
            @warn "Failed to converge within $GN_MAX_ITS iterations."
            return GNResult(false, θ, u, p, Au, ∂Ap∂u, n_solves)
        end

    end

end

function compute_local_lin(
    g::Grid,
    m::Model,
    pr::MaternField,
    y::AbstractVector,
    θ0::AbstractVector,
    C_e_inv::AbstractMatrix;
    n_eigvals::Int=30
)

    map = compute_map(g, m, pr, y, θ0, C_e_inv)
    !map.converged && @warn "MAP optimisation failed to converge."

    f(x) = compute_Hmx(x, map.Au, map.∂Ap∂u, g, m, pr, C_e_inv)

    vals, vecs, info = eigsolve(f, pr.Nω, n_eigvals, :LM, issymmetric=true)
    info.converged != length(vals) && @warn "eigsolve did not converge."
    
    λ_r = vals
    V_r = hcat(vecs...)

    D_r = diagm(λ_r ./ (λ_r .+ 1.0))
    P_r = diagm(1.0 ./ sqrt.(λ_r .+ 1.0) .- 1.0)

    C_post = I - V_r * D_r * V_r'
    L_post = V_r * P_r * V_r' + I

    n_solves = map.n_solves + 2 * info.numops

    @info "Minimum eigenvalue: $(minimum(vals))"
    @info "Total solves: $(n_solves)"
    
    return map, C_post, L_post, n_solves

end