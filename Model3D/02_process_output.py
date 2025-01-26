from enum import Enum

from setup import *


class ExitFlag(Enum):
    SUCCESS = 1
    FAILURE = 2


def get_status(ns_path, pr_path):
    """Returns a flag that indicates whether a combined NS and PR 
    simulation was successful.
    """
    
    if (flag := get_exitflag(ns_path)) == ExitFlag.FAILURE: 
        return flag
    
    return get_exitflag(pr_path)


def get_exitflag(log_path):
    """Determines the outcome of an NS or PR simulation."""

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


if __name__ == "__main__":
    
    Ne = 2000

    Fs = []
    Gs = []

    for i in range(Ne):
        
        ns_path = f"models/FL8788_{i}_NS"
        pr_path = f"models/FL8788_{i}_PR"

        flag = get_status(ns_path, pr_path)
        print(f"{i}: {flag}")

        if flag == ExitFlag.SUCCESS:
            
            F_i, G_i = data_handler_crse.get_pr_data(pr_path)
            ps = data_handler_crse.get_pr_pressures(F_i)

            if np.min(ps) > P_ATM / 1e6:
                Fs.append(F_i)
                Gs.append(G_i)
            
            else: 
                utils.warn("Simulation contains very low pressures.")

    Fs = np.hstack([F_i[:, np.newaxis] for F_i in Fs])
    Gs = np.hstack([G_i[:, np.newaxis] for G_i in Gs])

    np.save("data/Fs", Fs)
    np.save("data/Gs", Gs)