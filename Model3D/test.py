import h5py
import numpy as np


for i in range(1000):
    with h5py.File(f"models/FL8788_{i}_NS.h5") as f:

        max_sat = np.max(f["cell_fields"]["fluid_vapour_saturation"][:])
        print(max_sat)