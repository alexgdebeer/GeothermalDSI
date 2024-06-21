import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from src.consts import *
from src.models import *


class DataHandler():

    def __init__(self, mesh: Mesh, wells, num_wells, num_cur_wells, 
                 cs_temp_obs, ts_prod_obs, tmax, nt):

        self.mesh = mesh

        self.wells = wells
        self.num_wells = num_wells
        self.num_cur_wells = num_cur_wells

        self.feedzone_coords = np.array([w.feedzone_cell.centre for w in wells])
        self.feedzone_inds = np.array([w.feedzone_cell.index for w in wells])

        self.tmax = tmax 
        self.ts = np.linspace(0, tmax, nt+1)
        self.nt = nt

        self.cs_temp_obs = cs_temp_obs
        self.ts_prod_obs = ts_prod_obs

        self.inds_prod_obs = np.searchsorted(self.ts, ts_prod_obs-EPS)
        self.num_ts_prod_obs = len(self.ts_prod_obs)

        self.ind_pr_temp = self.inds_prod_obs[-1]

        self.downhole_coords = []
        self.downhole_elevs = []
        num_downhole_preds = []

        for well in self.wells:
            
            elevs = [l.centre for l in self.mesh.m.layer 
                     if well.min_elev <= l.centre <= well.max_elev]
            
            if well.min_elev not in elevs:
                elevs.append(well.min_elev)
            
            coords = [[well.x, well.y, e] for e in elevs]

            self.downhole_elevs.append(elevs)
            self.downhole_coords.extend(coords)
            num_downhole_preds.append(len(coords))

        self.cum_downhole_preds = np.cumsum([0, *num_downhole_preds])

        self.num_ns_temp = len(self.downhole_coords)
        self.num_pr_temp = len(self.downhole_coords)
        self.num_pr_pres = (self.nt+1) * self.num_wells
        self.num_pr_enth = (self.nt+1) * self.num_wells

        self.num_temp_obs = len(self.cs_temp_obs)
        self.num_pres_obs = self.num_ts_prod_obs * self.num_cur_wells
        self.num_enth_obs = self.num_ts_prod_obs * self.num_cur_wells

        self.generate_inds_preds()
        self.generate_inds_obs()

    def generate_inds_preds(self):
        """Generates indices used to extract temperatures, pressures 
        and enthalpies from a vector of complete data.
        """
        
        self.inds_ns_temp = np.arange(self.num_ns_temp)
        self.inds_pr_temp = np.arange(self.num_pr_temp) + 1 + self.inds_ns_temp[-1]
        self.inds_pr_pres = np.arange(self.num_pr_pres) + 1 + self.inds_pr_temp[-1]
        self.inds_pr_enth = np.arange(self.num_pr_enth) + 1 + self.inds_pr_pres[-1]

    def generate_inds_obs(self):
        """Generates indices used to extract temperatures, pressures 
        and enthalpy observations from a vector of observations.
        """
        
        self.inds_obs_temp = np.arange(self.num_temp_obs)
        self.inds_obs_pres = np.arange(self.num_pres_obs) + 1 + self.inds_obs_temp[-1]
        self.inds_obs_enth = np.arange(self.num_enth_obs) + 1 + self.inds_obs_pres[-1]

    def get_pr_data(self, pr_path):
        """Returns the temperatures (deg C), pressures (MPa) and
        enthalpies (kJ/kg) from a production history simulation.
        """

        with h5py.File(f"{pr_path}.h5", "r") as f:
        
            cell_inds = f["cell_index"][:, 0]
            src_inds = f["source_index"][:, 0]
            
            temp = f["cell_fields"]["fluid_temperature"]
            pres = f["cell_fields"]["fluid_pressure"]
            enth = f["source_fields"]["source_enthalpy"]

            ns_temp = np.array(temp[0][cell_inds])
            pr_temp = np.array(temp[self.ind_pr_temp][cell_inds])

            ns_interp = LinearNDInterpolator(self.mesh.tri, ns_temp)
            pr_interp = LinearNDInterpolator(self.mesh.tri, pr_temp)

            ns_temp = ns_interp(self.downhole_coords)
            pr_temp = pr_interp(self.downhole_coords)

            pr_pres = [p[cell_inds][self.feedzone_inds] for p in pres]
            pr_enth = [e[src_inds][-self.num_wells:] for e in enth]

            pr_pres = np.array(pr_pres) / 1e6
            pr_enth = np.array(pr_enth) / 1e3

            F_i = np.concatenate((ns_temp, pr_temp,
                                  pr_pres.flatten(), 
                                  pr_enth.flatten()))

            ns_temp_obs = ns_interp(self.cs_temp_obs)
            pr_pres_obs = pr_pres[self.inds_prod_obs, :self.num_cur_wells]
            pr_enth_obs = pr_enth[self.inds_prod_obs, :self.num_cur_wells]
        
            G_i = np.concatenate((ns_temp_obs,
                                  pr_pres_obs.flatten(),
                                  pr_enth_obs.flatten()))

        return F_i, G_i

    def reshape_to_wells(self, preds):
        """Reshapes a 1D array of predictions such that each column 
        contains the predictions for a single well.
        """
        
        return np.reshape(preds, (-1, self.num_wells))
    
    def reshape_to_cur_wells(self, preds):
        """Reshapes a 1D array of predictions such that each column 
        contains the predictions for a single well.
        """
        
        return np.reshape(preds, (-1, self.num_cur_wells))
    
    def group_by_well(self, preds):
        """Groups a set of downhole predictions by well."""
        
        preds_grouped = []
        for i0, i1 in zip(self.cum_downhole_preds[:-1], 
                          self.cum_downhole_preds[1:]):
            preds_i = preds[i0:i1]
            preds_grouped.append(preds_i)
        
        return preds_grouped
    
    def get_ns_temperatures(self, F_i):
        temp = F_i[self.inds_ns_temp]
        return self.group_by_well(temp)

    def get_pr_temperatures(self, F_i):
        temp = F_i[self.inds_pr_temp]
        return self.group_by_well(temp)

    def get_pr_pressures(self, F_i):
        pres = F_i[self.inds_pr_pres]
        return self.reshape_to_wells(pres)

    def get_pr_enthalpies(self, F_i):
        enth = F_i[self.inds_pr_enth]
        return self.reshape_to_wells(enth)

    def get_full_states(self, F_i):
        ns_temp = self.get_ns_temperatures(F_i)
        pr_temp = self.get_pr_temperatures(F_i)
        pr_pres = self.get_pr_pressures(F_i)
        pr_enth = self.get_pr_enthalpies(F_i)
        return ns_temp, pr_temp, pr_pres, pr_enth 

    def split_obs(self, G_i):
        """Splits a set of observations into temperatures, pressures 
        and enthalpies.
        """
        
        temp_obs = self.reshape_to_cur_wells(G_i[self.inds_obs_temp])
        pres_obs = self.reshape_to_cur_wells(G_i[self.inds_obs_pres])
        enth_obs = self.reshape_to_cur_wells(G_i[self.inds_obs_enth])
        return temp_obs, pres_obs, enth_obs
    
    def log_transform_pressures(self, F_i, eps=0.01):

        pred_ind_0 = self.inds_prod_obs[-1]
        pres = self.get_pr_pressures(F_i)

        n_pres, n_wells = pres.shape

        for i in range(n_pres-1, pred_ind_0, -1):
            for j in range(n_wells):
                dp = pres[i-1, j] - pres[i, j]
                pres[i, j] = np.log(dp + eps)
        
        F_i[self.inds_pr_pres] = pres.flatten()
        return F_i

    def inv_transform_pressures(self, F_i, eps=0.01):

        pred_ind_0 = self.inds_prod_obs[-1]
        pres = self.get_pr_pressures(F_i)

        n_pres, n_wells = pres.shape

        for i in range(pred_ind_0 + 1, n_pres):
            for j in range(n_wells):
                pres[i, j] = pres[i-1, j] - np.exp(pres[i, j]) + eps
        
        F_i[self.inds_pr_pres] = pres.flatten()
        return F_i