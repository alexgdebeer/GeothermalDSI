using Distributions
using HDF5
using LinearAlgebra
using Random: seed!

include("DarcyFlow/DarcyFlow.jl")

seed!(16)

FILE_TRUTH = "data/truth.h5"

# ----------------
# Reservoir properties 
# ----------------

μ = 0.5 * 1e-3 / (3600.0 * 24.0)    # Viscosity (Pa⋅day)
ϕ = 0.30                            # Porosity
c = 2.0e-4 / 6895.0                 # Compressibility (Pa^-1)
p0 = 20 * 1.0e6                     # Initial pressure (Pa)

# ----------------
# Grid
# ----------------

xmax = 1000.0
tmax = 160.0

Δx_c = 12.5
Δx_f = 7.5
Δt_c = 4.0
Δt_f = 2.0

n_wells = 9
well_centres = [
    (150, 150), (150, 500), (150, 850),
    (500, 150), (500, 500), (500, 850),
    (850, 150), (850, 500), (850, 850)
]

x_wells = [c[1] for c ∈ well_centres]
y_wells = [c[2] for c ∈ well_centres]
t_obs = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
t_preds = 84:4:160

grid_c = Grid(xmax, tmax, Δx_c, Δt_c)
grid_f = Grid(xmax, tmax, Δx_f, Δt_f)

# ----------------
# Well parameters 
# ----------------

q_c = 50.0 / Δx_c^2                 # Extraction rate, (m^3 / day) / m^3
q_f = 50.0 / Δx_f^2                 # Extraction rate, (m^3 / day) / m^3

well_radius = 50.0
well_change_times = [0, 40, 80, 120]

well_rates_c = [
    (-q_c, 0, 0, -0.5q_c), (0, -q_c, 0, -0.5q_c), (-q_c, 0, 0, -0.5q_c),
    (0, -q_c, 0, -0.5q_c), (-q_c, 0, 0, -0.5q_c), (0, -q_c, 0, -0.5q_c),
    (-q_c, 0, 0, -0.5q_c), (0, -q_c, 0, -0.5q_c), (-q_c, 0, 0, -0.5q_c)
]

well_rates_f = [
    (-q_f, 0, 0, -0.5q_f), (0, -q_f, 0, -0.5q_f), (-q_f, 0, 0, -0.5q_f),
    (0, -q_f, 0, -0.5q_f), (-q_f, 0, 0, -0.5q_f), (0, -q_f, 0, -0.5q_f),
    (-q_f, 0, 0, -0.5q_f), (0, -q_f, 0, -0.5q_f), (-q_f, 0, 0, -0.5q_f)
]

wells_c = [
    Well(grid_c, centre..., well_radius, rates) 
    for (centre, rates) ∈ zip(well_centres, well_rates_c)
]

wells_f = [
    Well(grid_f, centre..., well_radius, rates) 
    for (centre, rates) ∈ zip(well_centres, well_rates_f)
]

model_f = Model(
    grid_f, ϕ, μ, c, p0, 
    wells_f, well_change_times, 
    x_wells, y_wells, t_obs, t_preds
)

model_c = Model(
    grid_c, ϕ, μ, c, p0, 
    wells_c, well_change_times, 
    x_wells, y_wells, t_obs, t_preds
)

function generate_truth(
    g::Grid,
    m::Model,
    μ::Real,
    σ::Real,
    l::Real
)

    true_field = MaternField(g, μ, σ, l)
    θ_t = vec(rand(true_field))
    u_t = transform(true_field, θ_t)
    F_t = solve(g, m, u_t)
    G_t = m.B_obs * F_t

    h5write(FILE_TRUTH, "θ_t", θ_t)
    h5write(FILE_TRUTH, "u_t", u_t)
    h5write(FILE_TRUTH, "F_t", F_t)
    h5write(FILE_TRUTH, "G_t", G_t)

    return θ_t, u_t, F_t, G_t

end

function generate_obs(
    G_t::AbstractVector, 
    C_ϵ::AbstractMatrix
)

    d_obs = G_t + rand(MvNormal(C_ϵ)) 
    h5write(FILE_TRUTH, "d_obs", d_obs)

    return d_obs

end

function read_truth()

    f = h5open(FILE_TRUTH)
    θ_t = f["θ_t"][:]
    u_t = f["u_t"][:]
    F_t = f["F_t"][:]
    G_t = f["G_t"][:]
    close(f)

    return θ_t, u_t, F_t, G_t

end

function read_obs()

    f = h5open(FILE_TRUTH)
    d_obs = f["d_obs"][:]
    close(f)

    return d_obs

end

# ----------------
# Prior and error distribution
# ----------------

μ_lnk = -31.0
σ_lnk = 1.0
l_lnk = 400.0

pr = MaternField(grid_c, μ_lnk, σ_lnk, l_lnk)

σ_e = p0 * 0.01
μ_e = zeros(model_f.n_obs)
C_e = diagm(fill(σ_e^2, model_f.n_obs))
C_e_inv = spdiagm(fill(σ_e^-2, model_f.n_obs))
e_dist = MvNormal(μ_e, C_e)

# ----------------
# Truth and observations
# ----------------

# θ_t, u_t, F_t, G_t = generate_truth(grid_f, model_f, μ_lnk, σ_lnk, l_lnk)
θ_t, u_t, F_t, G_t = read_truth()

# d_obs = generate_obs(G_t, C_e)
d_obs = read_obs()

# ----------------
# POD
# ----------------

# Generate POD basis 
# μ_pi, V_ri = generate_pod_data(grid_c, model_c, pr, 100, 0.999, "pod/grid_$(grid_c.nx)")
μ_pi, V_ri = read_pod_data("pod/grid_$(grid_c.nx)")

model_r = ReducedOrderModel(
    grid_c, ϕ, μ, c, p0, wells_c, well_change_times,
    x_wells, y_wells, t_obs, t_preds, μ_pi, V_ri, μ_e, C_e
)

# ----------------
# Model functions
# ----------------

F(u::AbstractVector) = solve(grid_c, model_r, u)
G(p::AbstractVector) = model_c.B_obs * p