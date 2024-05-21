using SparseArrays

"""Well that has been mollified using a bump function of radius r."""
struct Well

    cx::Real
    cy::Real
    r::Real

    rates::Tuple
    Z::Real
    
    function Well(
        g::Grid, 
        cx::Real, 
        cy::Real, 
        r::Real,
        rates::Tuple
    )::Well

        """Calculates the value to scale the bump function by, such that 
        the values of the function on the model grid sum to 1."""
        function normalising_constant(
            g::Grid, 
            cx::Real, 
            cy::Real, 
            r::Real
        )::Real
        
            Z = 0.0
            for (x, y) ∈ zip(g.cxs, g.cys)
                if (r_sq = (x-cx)^2 + (y-cy)^2) < r^2
                    Z += exp(-1/(r^2-r_sq))
                end
            end
        
            return Z
        
        end

        Z = normalising_constant(g, cx, cy, r)

        return new(cx, cy, r, rates, Z)
    
    end

end

abstract type AbstractModel end

struct Model <: AbstractModel

    ϕ::Real
    μ::Real
    c::Real
    p0::Real

    Q::SparseMatrixCSC
    B::SparseMatrixCSC
    B_wells::SparseMatrixCSC

    ny::Int
    nyi::Int

    function Model(
        g::Grid,
        ϕ::Real, 
        μ::Real, 
        c::Real, 
        p0::Real,
        wells::AbstractVector{Well},
        well_change_times::AbstractVector,
        x_obs::AbstractVector,
        y_obs::AbstractVector,
        t_obs::AbstractVector
    )

        nyi = length(x_obs)
        ny = nyi * length(t_obs)

        t_obs_inds = [findfirst(g.ts .>= t) for t ∈ t_obs]

        Q = build_Q(g, wells, well_change_times)
        B, B_wells = build_B(g, ny, nyi, x_obs, y_obs, t_obs_inds)

        return new(ϕ, μ, c, p0, Q, B, B_wells, ny, nyi)

    end

end

struct ReducedOrderModel <: AbstractModel

    ϕ::Real
    μ::Real
    c::Real
    p0::Real

    Q::SparseMatrixCSC
    B::SparseMatrixCSC
    B_wells::SparseMatrixCSC
    BV_r::SparseMatrixCSC
    P::SparseMatrixCSC

    μ_pi::AbstractVector
    V_ri::AbstractMatrix
    
    μ_e::AbstractVector
    C_e::AbstractMatrix 
    C_e_inv::AbstractMatrix
    L_e::AbstractMatrix

    np_r::Int
    n_obs::Int
    n_pred::Int
    nyi::Int

    function ReducedOrderModel(
        g::Grid,
        ϕ::Real, 
        μ::Real, 
        c::Real, 
        p0::Real,
        wells::AbstractVector{Well},
        well_change_times::AbstractVector,
        x_obs::AbstractVector,
        y_obs::AbstractVector,
        t_obs::AbstractVector,
        t_pred::AbstractVector,
        μ_pi::AbstractVector,
        V_ri::AbstractMatrix,
        μ_e::AbstractVector,
        C_e::AbstractMatrix
    )

        np_r = size(V_ri, 2)
        nyi = length(x_obs)
        n_obs = nyi * length(t_obs)
        n_pred = nyi * length(t_pred)

        t_obs_inds = [findfirst(g.ts .>= t-1e-8) for t ∈ t_obs]
        t_pred_inds = [findfirst(g.ts .>= t-1e-8) for t ∈ t_pred]

        Q = build_Q(g, wells, well_change_times)
        B, B_wells = build_B(g, n_obs, nyi, x_obs, y_obs, t_obs_inds)
        P = build_P(g, n_pred, nyi, x_obs, y_obs, t_pred_inds)

        V_r = sparse(kron(sparse(I, g.nt, g.nt), V_ri))
        BV_r = B * V_r

        C_e_inv = Hermitian(inv(C_e))
        L_e = cholesky(C_e_inv).U

        return new(
            ϕ, μ, c, p0, 
            Q, B, B_wells, BV_r, P, 
            μ_pi, V_ri, 
            μ_e, C_e, C_e_inv, L_e,
            np_r, n_obs, n_pred, nyi
        )

    end

end

"""Builds the observation operator."""
function build_B(
    g::Grid,
    ny::Int,
    nyi::Int,
    x_obs::AbstractVector,
    y_obs::AbstractVector,
    t_obs_inds::AbstractVector 
)::Tuple{SparseMatrixCSC, SparseMatrixCSC}

    function get_cell_index(xi::Int, yi::Int)
        return xi + g.nx * (yi-1)
    end

    is = Int[]
    js = Int[]
    vs = Float64[]

    for (i, (x, y)) ∈ enumerate(zip(x_obs, y_obs))

        ix0 = findfirst(g.xs .> x) - 1
        iy0 = findfirst(g.xs .> y) - 1
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        x0, x1 = g.xs[ix0], g.xs[ix1]
        y0, y1 = g.xs[iy0], g.xs[iy1]

        inds = [(ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1)]
        cell_inds = [get_cell_index(i...) for i ∈ inds]

        Z = (x1-x0) * (y1-y0)

        push!(is, i, i, i, i)
        push!(js, cell_inds...)
        push!(vs,
            (x1-x) * (y1-y) / Z, 
            (x1-x) * (y-y0) / Z, 
            (x-x0) * (y1-y) / Z, 
            (x-x0) * (y-y0) / Z
        )

    end

    B = spzeros(ny, g.nx^2 * g.nt)
    Bi = sparse(is, js, vs, nyi, g.nx^2)

    for (i, t) ∈ enumerate(t_obs_inds)
        ii = (i-1) * nyi
        jj = (t-1) * g.nx^2
        B[(ii+1):(ii+nyi), (jj+1):(jj+g.nx^2)] = Bi
    end

    B_wells = blockdiag([Bi for _ ∈ 1:g.nt]...)

    return B, B_wells

end

"""Builds the operator that extracts the predictive quantities of 
interest from a set of model output."""
function build_P(
    g::Grid,
    ny::Int,
    nyi::Int,
    x_pred::AbstractVector,
    y_pred::AbstractVector,
    t_pred_inds::AbstractVector 
)::SparseMatrixCSC

    function get_cell_index(xi::Int, yi::Int)
        return xi + g.nx * (yi-1)
    end

    is = Int[]
    js = Int[]
    vs = Float64[]

    for (i, (x, y)) ∈ enumerate(zip(x_pred, y_pred))

        ix0 = findfirst(g.xs .> x) - 1
        iy0 = findfirst(g.xs .> y) - 1
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        x0, x1 = g.xs[ix0], g.xs[ix1]
        y0, y1 = g.xs[iy0], g.xs[iy1]

        inds = [(ix0, iy0), (ix0, iy1), (ix1, iy0), (ix1, iy1)]
        cell_inds = [get_cell_index(i...) for i ∈ inds]

        Z = (x1-x0) * (y1-y0)

        push!(is, i, i, i, i)
        push!(js, cell_inds...)
        push!(vs,
            (x1-x) * (y1-y) / Z, 
            (x1-x) * (y-y0) / Z, 
            (x-x0) * (y1-y) / Z, 
            (x-x0) * (y-y0) / Z
        )

    end

    P = spzeros(ny, g.nx^2 * g.nt)
    Pi = sparse(is, js, vs, nyi, g.nx^2)

    for (i, t) ∈ enumerate(t_pred_inds)
        ii = (i-1) * nyi
        jj = (t-1) * g.nx^2
        P[(ii+1):(ii+nyi), (jj+1):(jj+g.nx^2)] = Pi
    end

    return P

end

"""Builds the forcing term at each time index."""
function build_Q(
    g::Grid,
    wells::AbstractVector{Well},
    well_change_times::AbstractVector 
)::SparseMatrixCSC

    Q_i = Int[]
    Q_j = Int[]
    Q_v = Float64[]

    time_inds = [findlast(well_change_times .<= t + 1e-8) for t ∈ g.ts]

    for (i, (x, y)) ∈ enumerate(zip(g.cxs, g.cys))
        for w ∈ wells 
            if (dist_sq = (x-w.cx)^2 + (y-w.cy)^2) < w.r^2
                for (j, q) ∈ enumerate(w.rates[time_inds])
                    push!(Q_i, i)
                    push!(Q_j, j)
                    push!(Q_v, q * exp(-1/(q^2-dist_sq)) / w.Z)
                end
            end
        end
    end

    Q = sparse(Q_i, Q_j, Q_v, g.nx^2, g.nt)
    return Q

end