include("setup.jl")

# predicting the pressures at each well after the end of the production period.

Ne = 1000

θs = rand(pr, Ne)
us = hcat([transform(pr, θ) for θ ∈ eachcol(θs)]...)

ps = hcat([F(u) for u ∈ eachcol(us)]...)
ds = model_c.B * ps
