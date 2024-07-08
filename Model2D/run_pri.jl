include("setup.jl")
include("InferenceAlgorithms/InferenceAlgorithms.jl")


u_pri = hcat([transform(pr, rand(pr, 1)) for _ ∈ 1:1000]...)
p_pri = hcat([F(u) for u ∈ eachcol(u_pri)]...)

p_pri_wells = [hcat(
    [reshape(p, 9, :)[i, :] 
    for p ∈ eachcol(model_c.B_wells * p_pri)]...
) for i in 1:9]

p_pri_wells = [vcat(fill(p0, 1000)', p) for p ∈ p_pri_wells]
p_pri_wells = cat(p_pri_wells...; dims=3)

h5write("data/results.h5", "pri_preds", p_pri_wells)
h5write("data/results.h5", "ts_preds", [0, grid_f.ts...])