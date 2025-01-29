include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")


Z_MIN = 1e-4


struct CDFTransformer

    dim_x::Int
    n_samples::Int
    
    FGs::AbstractMatrix
    FGs_sorted::AbstractMatrix

    cdf_elem_size::AbstractFloat
    row_inds::UnitRange

    function CDFTransformer(
        FGs::AbstractMatrix, 
        FGs_min::AbstractVector, 
        FGs_max::AbstractVector
    )

        FGs = hcat(FGs, FGs_min, FGs_max)
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
    
    mask_l = zs .< Z_MIN
    mask_r = zs .> 1.0 .- Z_MIN
    mask_c = .!(mask_l) .& .!(mask_r)

    # Rescale stuff to account for less probability at the edges
    zs[mask_l] .*= (Z_MIN ./ t.cdf_elem_size)
    zs[mask_r] = 1.0 .- Z_MIN * (1.0 .- zs[mask_r]) ./ t.cdf_elem_size
    zs[mask_c] = Z_MIN .+ (zs[mask_c] .- t.cdf_elem_size) .* (1.0 .- 2.0 .* Z_MIN) ./ (1.0 .- 2.0 .* t.cdf_elem_size)
    
    clamp!(zs, 1e-8, 1.0-1e-8)

    ξs = quantile(Normal(), zs)
    return ξs

end

function transform(
    t::CDFTransformer, 
    ξs_i::AbstractVector
)

    zs_i = cdf.(Normal(), ξs_i)
    clamp!(zs_i, 1e-8, 1.0-1e-8)

    # Rescale elements of CDF
    mask_l = zs_i .< Z_MIN
    mask_r = zs_i .> 1.0 .- Z_MIN
    mask_c = .!(mask_l) .& .!(mask_r)

    zs_i[mask_l] .*= (t.cdf_elem_size ./ Z_MIN)
    zs_i[mask_r] = 1.0 .- t.cdf_elem_size * (1.0 .- zs_i[mask_r]) ./ Z_MIN 
    zs_i[mask_c] = t.cdf_elem_size .+ (zs_i[mask_c] .- Z_MIN) .* (1.0 .- 2.0*t.cdf_elem_size) ./ (1.0 .- 2.0*Z_MIN)

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
    FGs_min = zeros(size(FGs, 1))
    FGs_max = 21 * 1.0e6 * ones(size(FGs, 1))

    t = CDFTransformer(FGs, FGs_min, FGs_max)
    
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

    # Fs_pri_wells = map_to_wells(Fs_pri)
    Fs_post_wells = map_to_wells(Fs_post)

    h5write("data/results_dsi.h5", "dsi_preds_$i", Fs_post_wells)

end

Fs_pri, _ = run_dsi(y_obs[:, 1:1000], y_pred[:, 1:1000], d_obs, n_samples)
Fs_pri = map_to_wells(Fs_pri)
h5write("data/results_dsi.h5", "dsi_pri", Fs_pri)