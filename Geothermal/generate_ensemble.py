from setup import *


Ne = 1000

ps = np.stack([prior.sample() for _ in range(Ne)], axis=1)
np.save("ps", ps)

for i, p_i in enumerate(ps.T):
    generate_particle(p_i, i)