from src.models import Ensemble

from setup import *


Ne = 1000

ps = np.stack([prior.sample() for _ in range(Ne)], axis=1)

ens = Ensemble(prior, generate_particle, get_result, NF, NG, Ne)
ens.generate(ps)

np.save("ps", ps)