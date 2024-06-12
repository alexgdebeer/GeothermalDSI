import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from src.consts import *
from src.models import *


class DataHandler():

    def __init__(self, mesh: Mesh, wells, n_wells, n_cur_wells, 
                 temp_obs_cs, prod_obs_ts, tmax, nt):

        self.mesh = mesh
        self.cell_cs = np.array([c.centre for c in mesh.m.cell])

        self.wells = wells
        self.n_wells = n_wells
        self.n_cur_wells = n_cur_wells

        self.feedzone_coords = np.array([w.feedzone_cell.centre for w in wells])
        self.feedzone_inds = np.array([w.feedzone_cell.index for w in wells])

        self.tmax = tmax 
        self.ts = np.linspace(0, tmax, nt+1)
        self.nt = nt

        self.temp_obs_cs = temp_obs_cs
        self.prod_obs_ts = prod_obs_ts

        self.inds_prod_obs = np.searchsorted(self.ts, prod_obs_ts-EPS)
        self.n_prod_obs_ts = len(self.prod_obs_ts)

        self.pr_temp_ind = self.inds_prod_obs[-1] # TODO: check if this is good.

        self.n_ns_temp = self.mesh.m.num_cells
        self.n_pr_temp = self.mesh.m.num_cells
        self.n_pr_pres = (self.nt+1) * self.n_wells
        self.n_pr_enth = (self.nt+1) * self.n_wells

        self.n_temp_obs = len(self.temp_obs_cs)
        self.n_pres_obs = self.n_prod_obs_ts * self.n_cur_wells
        self.n_enth_obs = self.n_prod_obs_ts * self.n_cur_wells

        self.generate_inds_full()
        self.generate_inds_obs()

    def generate_inds_full(self):
        """Generates indices used to extract temperatures, pressures 
        and enthalpies from a vector of complete data.
        """
        
        self.inds_ns_temp = np.arange(self.n_ns_temp)
        self.inds_pr_temp = np.arange(self.n_pr_temp) + 1 + self.inds_ns_temp[-1]
        self.inds_pr_pres = np.arange(self.n_pr_pres) + 1 + self.inds_pr_temp[-1]
        self.inds_pr_enth = np.arange(self.n_pr_enth) + 1 + self.inds_pr_pres[-1]

    def generate_inds_obs(self):
        """Generates indices used to extract temperatures, pressures 
        and enthalpy observations from a vector of observations.
        """
        
        self.inds_obs_temp = np.arange(self.n_temp_obs)
        self.inds_obs_pres = np.arange(self.n_pres_obs) + 1 + self.inds_obs_temp[-1]
        self.inds_obs_enth = np.arange(self.n_enth_obs) + 1 + self.inds_obs_pres[-1]

    def get_pr_data(self, pr_path):
        """Returns the temperatures (deg C), pressures (MPa) and
        enthalpies (kJ/kg) from a production history simulation."""

        with h5py.File(f"{pr_path}.h5", "r") as f:
        
            cell_inds = f["cell_index"][:, 0]
            src_inds = f["source_index"][:, 0]
            
            temp = f["cell_fields"]["fluid_temperature"]
            pres = f["cell_fields"]["fluid_pressure"]
            enth = f["source_fields"]["source_enthalpy"]

            ns_temp = np.array(temp[0][cell_inds])
            pr_temp = np.array(temp[self.pr_temp_ind][cell_inds])

            pr_pres = [p[cell_inds][self.feedzone_inds] for p in pres]
            pr_enth = [e[src_inds][-self.n_wells:] for e in enth]

            pr_pres = np.array(pr_pres) / 1e6
            pr_enth = np.array(pr_enth) / 1e3

        F_i = np.concatenate((ns_temp.flatten(), 
                              pr_temp.flatten(),
                              pr_pres.flatten(), 
                              pr_enth.flatten()))

        return F_i

    def reshape_to_wells(self, obs):
        """Reshapes a 1D array of observations such that each column 
        contains the observations for a single well.
        """
        
        return np.reshape(obs, (-1, self.n_wells))
    
    def reshape_to_cur_wells(self, obs):
        """Reshapes a 1D array of observations such that each column 
        contains the observations for a single well.
        """
        
        return np.reshape(obs, (-1, self.n_cur_wells))
    
    def get_full_ns_temperatures(self, F_i):
        temp = F_i[self.inds_ns_temp]
        return temp

    def get_full_temperatures(self, F_i):
        temp = F_i[self.inds_pr_temp]
        return temp

    def get_full_pressures(self, F_i):
        pres = F_i[self.inds_pr_pres]
        return self.reshape_to_wells(pres)

    def get_full_enthalpies(self, F_i):
        enth = F_i[self.inds_pr_enth]
        return self.reshape_to_wells(enth)

    def get_full_states(self, F_i):
        ns_temp = self.get_full_ns_temperatures(F_i)
        temp = self.get_full_temperatures(F_i)
        pres = self.get_full_pressures(F_i)
        enth = self.get_full_enthalpies(F_i)
        return ns_temp, temp, pres, enth 
    
    def get_obs_temperatures(self, temp_full):
        """Extracts the temperatures at each observation point from a 
        full set of temperatures.
        """
        
        interp = LinearNDInterpolator(self.mesh.tri, temp_full)
        temp_obs = interp(self.temp_obs_cs)
        return self.reshape_to_cur_wells(temp_obs)
    
    def get_obs_pressures(self, pres_full):
        return pres_full[self.inds_prod_obs, :self.n_cur_wells]

    def get_obs_enthalpies(self, enth_full):
        return enth_full[self.inds_prod_obs, :self.n_cur_wells]
    
    def get_obs_states(self, F_i):
        """Extracts the observations from a complete vector of model 
        output, and returns each type of observation individually.
        """

        ns_temp_full, _, pres_full, enth_full = self.get_full_states(F_i)
        temp_obs = self.get_obs_temperatures(ns_temp_full)
        pres_obs = self.get_obs_pressures(pres_full)
        enth_obs = self.get_obs_enthalpies(enth_full)
        return temp_obs, pres_obs, enth_obs
    
    def get_obs(self, F_i):
        """Extracts the observations from a complete vector of model
        output, and returns them as a vector.
        """

        temp_obs, pres_obs, enth_obs = self.get_obs_states(F_i)
        obs = np.concatenate((temp_obs.flatten(), 
                              pres_obs.flatten(), 
                              enth_obs.flatten()))
        return obs

    def split_obs(self, G_i):
        """Splits a set of observations into temperatures, pressures 
        and enthalpies.
        """
        
        temp_obs = self.reshape_to_cur_wells(G_i[self.inds_obs_temp])
        pres_obs = self.reshape_to_cur_wells(G_i[self.inds_obs_pres])
        enth_obs = self.reshape_to_cur_wells(G_i[self.inds_obs_enth])
        return temp_obs, pres_obs, enth_obs

    def get_downhole_temps(self, temp_full, well_num):
        """Returns interpolated temperatures down a single well."""

        well = self.wells[well_num]
        elevs = [l.centre for l in self.mesh.m.layer
                 if well.min_elev <= l.centre <= well.max_elev]
        if well.min_elev not in elevs:
            elevs.append(well.min_elev)
        
        coords = np.array([[well.x, well.y, e] for e in elevs])
        interp = LinearNDInterpolator(self.mesh.tri, temp_full)
        downhole_temps = interp(coords)
        return elevs, downhole_temps
    
    def log_transform_pressures(self, F_i, eps=0.01):

        pred_ind_0 = self.inds_prod_obs[-1]
        pres = self.get_full_pressures(F_i)

        n_pres, n_wells = pres.shape

        for i in range(n_pres-1, pred_ind_0, -1):
            for j in range(n_wells):
                dp = pres[i-1, j] - pres[i, j]
                pres[i, j] = np.log(dp + eps)
        
        F_i[self.inds_pr_pres] = pres.flatten()
        return F_i

    def inv_transform_pressures(self, F_i, eps=0.01):

        pred_ind_0 = self.inds_prod_obs[-1]
        pres = self.get_full_pressures(F_i)

        n_pres, n_wells = pres.shape

        for i in range(pred_ind_0 + 1, n_pres):
            for j in range(n_wells):
                pres[i, j] = pres[i-1, j] - np.exp(pres[i, j]) + eps
        
        F_i[self.inds_pr_pres] = pres.flatten()
        return F_i