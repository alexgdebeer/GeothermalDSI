using HDF5
using Statistics

include("setup.jl")


N_CHAINS = 5
N_BATCHES = 5_000
N_SAMPLES_PER_BATCH = 1
N_SAMPLES = N_BATCHES * N_SAMPLES_PER_BATCH
WARMUP_LENGTH = 2_500 # Warmup length in batches

DATA_FOLDER = "data/pcn"


function compute_psrf(
    ωs::AbstractMatrix
)::Real

    # Split each chain in half 
    ωs = reshape(ωs, :, 2N_CHAINS)
    n, m = size(ωs)
    
    μ = mean(ωs)
    μ_js = [mean(c) for c ∈ eachcol(ωs)]
    s_js = [1 / (n-1) * sum((c .- μ_js[j]).^2) 
            for (j, c) ∈ enumerate(eachcol(ωs))]

    B = n / (m-1) * sum((μ_js .- μ).^2)
    W = 1 / m * sum(s_js)

    varp = (n-1)/n * W + 1/n * B
    psrf = sqrt(varp / W)

    return psrf

end

ωs = zeros(pr.Nω, N_SAMPLES, N_CHAINS)
us = zeros(pr.Nθ, N_SAMPLES, N_CHAINS)
τs = zeros(100N_BATCHES, N_CHAINS)

for i ∈ 1:N_CHAINS

    f = h5open("$(DATA_FOLDER)/chain_$i.h5", "r")
    ωs[:, :, i] = reduce(hcat, [f["ωs_$b"][:, 1] for b ∈ 1:N_BATCHES])
    us[:, :, i] = reduce(hcat, [f["us_$b"][:, 1] for b ∈ 1:N_BATCHES])
    τs[:, i] = reduce(vcat, [f["τs_$b"][:, 1] for b ∈ 1:N_BATCHES])
    close(f)
    @info "Finished reading data from chain $i."

end

psrfs_ω = [compute_psrf(ωs[i, WARMUP_LENGTH+1:end, :]) for i ∈ 1:pr.Nω]

println(psrfs_ω)

sample_inds = rand(WARMUP_LENGTH+1:N_SAMPLES, 200)
us_samp = reshape(us[:, sample_inds, :], 2500, 1000)

p_post_wells = [hcat(
    [reshape(p, 9, :)[i, :] 
    for p ∈ eachcol(model_c.B_wells * us_samp)]...
) for i in 1:9]

p_post_wells = [vcat(fill(p0, 1000)', p) for p ∈ p_post_wells]
p_post_wells = cat(p_post_wells...; dims=3)

h5write("data/results.h5", "pcn_preds", p_post_wells)