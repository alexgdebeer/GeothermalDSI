using Distributions


UNIT_NORM = Normal()


function run_chain(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    L_e::AbstractMatrix,
    θ0::AbstractVector,
    NF::Int,
    NG::Int,
    Ni::Int, 
    Nb::Int,
    β::Real,
    n_chain::Int;
    thin::Int=10,
    verbose::Bool=true
)

    logpri(θ) = -0.5sum(θ.^2)
    loglik(G) = -0.5sum((L_e*(G+μ_e-d_obs)).^2)
    logpost(θ, G) = logpri(θ) + loglik(G)

    θs = Matrix{Float64}(undef, pr.Nθ, Nb)
    us = Matrix{Float64}(undef, pr.Nθ, Nb)
    Fs = Matrix{Float64}(undef, NF, Nb)
    Gs = Matrix{Float64}(undef, NG, Nb)
    τs = Vector{Float64}(undef, Nb)

    α = 0

    u0 = transform(pr, θ0)

    θs[:, 1] = θ0
    us[:, 1] = u0

    Fs[:, 1] = F(u0)
    Gs[:, 1] = G(Fs[:, 1])
    τs[1] = logpost(θ0, Gs[:, 1])

    t0 = time()
    n_chunk = 1
    for i ∈ 1:(Ni-1)

        ind_c = (i-1) % Nb + 1
        ind_p = i % Nb + 1

        ζ = rand(UNIT_NORM, pr.Nθ)
        θ_p = √(1-β^2) * θs[:, ind_c] + β*ζ
        u_p = transform(pr, θ_p)
        F_p = F(u_p)
        G_p = G(F_p)

        h = exp(loglik(G_p) - loglik(Gs[:, ind_c]))

        if h ≥ rand()
            α += 1
            θs[:, ind_p] = θ_p
            us[:, ind_p] = u_p
            Fs[:, ind_p] = F_p
            Gs[:, ind_p] = G_p
        else
            θs[:, ind_p] = θs[:, ind_c]
            us[:, ind_p] = us[:, ind_c]
            Fs[:, ind_p] = Fs[:, ind_c]
            Gs[:, ind_p] = Gs[:, ind_c]
        end

        τs[ind_p] = logpost(θs[:, ind_p], Gs[:, ind_p])

        if (i+1) % Nb == 0

            h5write("data/pcn/chain_$n_chain.h5", "θs_$n_chunk", θs[:, 1:thin:end])
            h5write("data/pcn/chain_$n_chain.h5", "us_$n_chunk", us[:, 1:thin:end])
            h5write("data/pcn/chain_$n_chain.h5", "τs_$n_chunk", τs[:, 1:thin:end])

            n_chunk += 1

            if verbose

                t1 = time()
                time_per_it = (t1 - t0) / Nb
                t0 = t1

                @printf(
                    "%5.0i | %5.0e | %6.2f | %9.2e | %7.3f\n",
                    n_chain, i, α/i, τs[ind_p], time_per_it
                )

            end

        end

    end

    return nothing

end

function run_pcn(
    F::Function,
    G::Function,
    pr::MaternField,
    d_obs::AbstractVector,
    μ_e::AbstractVector,
    L_e::AbstractMatrix,
    NF::Int,
    Ni::Int,
    Nb::Int,
    Nc::Int,
    β::Real,
    δ::Real,
    verbose::Bool=true
)

    verbose && println("Chain | Iters | logpost   | time (s)")

    NG = length(d_obs)

    Threads.@threads for chain_num ∈ 1:Nc

        θ0 = vec(rand(pr))
        run_chain(
            F, G, pr, d_obs, μ_e, L_e, θ0, 
            NF, NG, Ni, Nb, β, δ,
            chain_num, verbose=verbose
        )

    end

end