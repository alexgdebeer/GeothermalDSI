from enum import Enum
import h5py 

from setup import *


class ExitFlag(Enum):
    SUCCESS = 1
    FAILURE = 2


def get_status(ns_path, pr_path):
    """Simulates the model and returns a flag that indicates 
    whether the simulation was successful."""
    
    if (flag := get_exitflag(ns_path)) == ExitFlag.FAILURE: 
        return flag
    
    return get_exitflag(pr_path)


def get_exitflag(log_path):
    """Determines the outcome of a simulation."""

    with open(f"{log_path}.yaml", "r") as f:
        log = yaml.safe_load(f)

    for msg in log[::-1]:

        if msg[:3] in [MSG_END_TIME, MSG_MAX_STEP]:
            return ExitFlag.SUCCESS

        elif msg[:3] == MSG_MAX_ITS:
            utils.warn("Simulation failed (max iterations).")
            return ExitFlag.FAILURE

        elif msg[:3] == MSG_ABORTED:
            utils.warn("Simulation failed (aborted).")
            return ExitFlag.FAILURE

    raise Exception(f"Unknown exit condition. Check {log_path}.yaml.")


def get_pr_data(pr_path, feedzone_cell_inds, n_feedzones):
    """Returns the temperatures (deg C), pressures (MPa) and
    enthalpies (kJ/kg) from a production history simulation."""

    with h5py.File(f"{pr_path}.h5", "r") as f:
    
        cell_inds = f["cell_index"][:, 0]
        src_inds = f["source_index"][:, 0]
        
        temp = f["cell_fields"]["fluid_temperature"]
        pres = f["cell_fields"]["fluid_pressure"]
        enth = f["source_fields"]["source_enthalpy"]

        ns_temp = np.array(temp[0][cell_inds])
        pr_pres = np.array([p[cell_inds][feedzone_cell_inds] for p in pres])
        pr_enth = np.array([e[src_inds][-n_feedzones:] for e in enth])

        pr_pres /= 1e6
        pr_enth /= 1e3

    F_i = np.concatenate((ns_temp.flatten(), 
                          pr_pres.flatten(), 
                          pr_enth.flatten()))

    return F_i

if __name__ == "__main__":
    
    Ne = 1000

    feedzone_cell_inds = np.array([w.feedzone_cell.index for w in wells_crse])
    n_feedzones = len(well_centres)

    Fs = []
    Gs = []

    for i in range(Ne):
        
        ns_path = f"models/FL8788_{i}_NS"
        pr_path = f"models/FL8788_{i}_PR"

        flag = get_status(ns_path, pr_path)
        print(f"{i}: {flag}")

        if flag == ExitFlag.SUCCESS:
            F_i = get_pr_data(pr_path, feedzone_cell_inds, n_feedzones)
            G_i = data_handler_crse.get_obs(F_i)
            Fs.append(F_i)
            Gs.append(G_i)

    Fs = np.hstack([F_i[:, np.newaxis] for F_i in Fs])
    Gs = np.hstack([G_i[:, np.newaxis] for G_i in Gs])

    np.save("data/Fs", Fs)
    np.save("data/Gs", Gs)