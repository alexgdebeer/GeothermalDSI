include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")


struct CDFTransformer

    dim_x::Int
    n_samples::Int
    
    FGs::AbstractMatrix
    FGs_sorted::AbstractMatrix

    cdf_elem_size::AbstractFloat
    row_inds::UnitRange

    function CDFTransformer(FGs::AbstractMatrix)

        dim_x, n_samples = size(FGs)
        FGs_sorted = sort(FGs, dims=2)

        cdf_elem_size = 1.0 / (n_samples - 1)
        row_inds = 1:dim_x

        return new(dim_x, n_samples, FGs, FGs_sorted, cdf_elem_size, row_inds)

    end

end

function transform_inv(
    t::CDFTransformer, 
    FG_i::AbstractVector
)

    inds_0 = sum(((FG_i .- t.FGs_sorted) .> -1e-8), dims=2)
    inds_0[inds_0 .== 0] .= 1
    inds_0[inds_0 .== t.n_samples] .= t.n_samples - 1
    inds_1 = inds_0 .+ 1

    FGs_0 = [t.FGs_sorted[ri, ci] for (ri, ci) in zip(t.row_inds, inds_0)]
    FGs_1 = [t.FGs_sorted[ri, ci] for (ri, ci) in zip(t.row_inds, inds_1)]

    dzs = (FG_i .- FGs_0) ./ (FGs_1 .- FGs_0)
    zs = t.cdf_elem_size * (inds_0 .+ dzs)
    clamp!(zs, 1e-4, 1.0-1e-4)
    ξs = quantile(Normal(), zs)
    return ξs

end

function transform(
    t::CDFTransformer, 
    ξs_i::AbstractVector
)

    zs_i = cdf.(Normal(), ξs_i)
    clamp!(zs_i, 1e-4, 1.0-1e-4)

    # Compute inverse CDF
    inds_0 = Int.(floor.(zs_i / t.cdf_elem_size))
    zs_loc = (zs_i .- inds_0 * t.cdf_elem_size) / t.cdf_elem_size

    FGs_0 = [t.FGs_sorted[ri, ci] for (ri, ci) in zip(t.row_inds, inds_0.+1)]
    FGs_1 = [t.FGs_sorted[ri, ci] for (ri, ci) in zip(t.row_inds, inds_0.+2)]
    FGs_i = FGs_0 .* (1.0 .- zs_loc) .+ FGs_1 .* zs_loc
    return FGs_i

end


function compute_chol(ξs::AbstractMatrix)

    C = cov(ξs')
    C += 1e-8I
    L = cholesky(C).L
    return L

end


function generate_prior_samples(
    t::CDFTransformer, 
    y_obs::AbstractVector, 
    L::AbstractMatrix,
    n::Int
)

    ny = length(y_obs)
    samples_η = rand(Normal(), t.dim_x, n)
    samples = hcat([transform(t, L * samples_η_i) for samples_η_i ∈ eachcol(samples_η)]...)
    return samples[ny+1:end, :]

end


function generate_conditional_samples(
    t::CDFTransformer, 
    y_obs::AbstractVector, 
    L::AbstractMatrix,
    n::Int
)

    L_inv = inv(L)

    # Compute inverse of y_obs
    ny = length(y_obs)
    y_obs_0 = [y_obs; zeros(t.dim_x - ny)]
    η_obs_0 = L_inv * transform_inv(t, y_obs_0)
    η_obs = η_obs_0[1:ny]
    
    # # Check transformations work...
    # n_obs_0 = np.hstack((n_obs.flatten(), np.zeros(self.dim_x-ny)))[:, None]
    # y_obs_1 = self.transform(n_obs_0).flatten()
    # # assert np.max(np.abs(y_obs[:ny]-y_obs_1[:ny])) < 1e-4, "Issue with transformations."

    samples_cond_η = rand(Normal(), t.dim_x, n)
    samples_cond_η[1:ny, :] .= η_obs
    samples_cond = hcat([transform(t, L * samples_cond_η_i) for samples_cond_η_i ∈ eachcol(samples_cond_η)]...)

    return samples_cond[ny+1:end, :]

end


function run_dsi(
    y_obs::AbstractMatrix, 
    y_pred::AbstractMatrix,
    d_obs::AbstractVector, 
    n_samples::Int
)

    FGs = [y_obs; y_pred]
    t = CDFTransformer(FGs)
    
    # Back-transform
    ξs = hcat([transform_inv(t, FGs_i) for FGs_i in eachcol(FGs)]...)
    L = compute_chol(ξs)

    Fs_pri = generate_prior_samples(t, d_obs, L, n_samples)
    Fs_post = generate_conditional_samples(t, d_obs, L, n_samples)
    return Fs_pri, Fs_post

end


function map_to_wells(ps::AbstractMatrix)

    ps_wells = [hcat(
        [reshape(p, 9, :)[i, :] for p ∈ eachcol(ps)]...
    ) for i in 1:9]

    ps_wells = [vcat(fill(p0, 1000)', p) for p ∈ ps_wells]
    ps_wells = cat(ps_wells...; dims=3)

    return ps_wells

end


Ne = 10_000

ωs = rand(pr, Ne)
us = hcat([transform(pr, ω) for ω ∈ eachcol(ωs)]...)

ps = hcat([@time F(u) for u ∈ eachcol(us)]...)
ϵs = rand(MvNormal(C_e), Ne)

y_obs = model_c.B_obs * ps + ϵs
y_pred = model_c.B_wells * ps

n_samples = 1000

for i ∈ [100, 250, 500, 1_000, 2_000, 5_000, 10_000]

    _, Fs_post = run_dsi(y_obs[:, 1:i], y_pred[:, 1:i], d_obs, n_samples)

    Fs_pri_wells = map_to_wells(Fs_pri)
    Fs_post_wells = map_to_wells(Fs_pri)

    h5write("data/results_dsi.h5", "dsi_preds_$i", p_post_wells)

end

Fs_pri, _ = run_dsi(y_obs, y_pred, d_obs, n_samples)
Fs_pri = map_to_wells(Fs_pri)
h5write("data/results_dsi.h5", "dsi_pri", Fs_pri)