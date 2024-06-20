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


struct Model

    ϕ::Real
    μ::Real
    c::Real
    p0::Real

    Q::SparseMatrixCSC
    B_obs::SparseMatrixCSC
    B_preds::SparseMatrixCSC
    B_wells::SparseMatrixCSC

    n_obs::Int
    n_preds::Int
    n_wells::Int

    function Model(
        g::Grid,
        ϕ::Real, 
        μ::Real, 
        c::Real, 
        p0::Real,
        wells::AbstractVector{Well},
        well_change_times::AbstractVector,
        x_wells::AbstractVector,
        y_wells::AbstractVector,
        t_obs::AbstractVector,
        t_preds::AbstractVector
    )

        n_wells = length(x_wells)
        n_obs = n_wells * length(t_obs)
        n_preds = n_wells * length(t_preds)

        t_obs_inds = [findfirst(g.ts .>= t-1e-8) for t ∈ t_obs]
        t_pred_inds = [findfirst(g.ts .>= t-1e-8) for t ∈ t_preds]

        Q = build_Q(g, wells, well_change_times)

        B_obs, B_preds, B_wells = build_Bs(
            g, n_wells, n_obs, n_preds, 
            x_wells, y_wells, t_obs_inds, t_pred_inds
        )

        return new(
            ϕ, μ, c, p0, Q, 
            B_obs, B_preds, B_wells, 
            n_obs, n_preds, n_wells
        )

    end

end


"""Builds operators that map between the full simulation outputs and
the observations and predictive QoIs."""
function build_Bs(
    g::Grid,
    n_wells::Int,
    n_obs::Int,
    n_preds::Int,
    x_wells::AbstractVector,
    y_wells::AbstractVector,
    t_obs_inds::AbstractVector,
    t_pred_inds::AbstractVector
)

    function get_cell_index(xi::Int, yi::Int)
        return xi + g.nx * (yi-1)
    end

    is = Int[]
    js = Int[]
    vs = Float64[]

    for (i, (x, y)) ∈ enumerate(zip(x_wells, y_wells))

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

    B_obs = spzeros(n_obs, g.nx^2 * g.nt)
    B_pred = spzeros(n_preds, g.nx^2 * g.nt)
    Bi = sparse(is, js, vs, n_wells, g.nx^2)

    for (i, t) ∈ enumerate(t_obs_inds)
        ii = (i-1) * n_wells
        jj = (t-1) * g.nx^2
        B_obs[(ii+1):(ii+n_wells), (jj+1):(jj+g.nx^2)] = Bi
    end

    for (i, t) ∈ enumerate(t_pred_inds)
        ii = (i-1) * n_wells
        jj = (t-1) * g.nx^2
        B_pred[(ii+1):(ii+n_wells), (jj+1):(jj+g.nx^2)] = Bi
    end

    B_wells = blockdiag([Bi for _ ∈ 1:g.nt]...)

    return B_obs, B_pred, B_wells

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