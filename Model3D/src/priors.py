import numpy as np

from src.models import *


class Prior():

    def __init__(self, mesh: Mesh, cap: ClayCap, 
                 fault: Fault, grf_ext: PermField, 
                 grf_flt: PermField, grf_cap: PermField, 
                 grf_upflow: UpflowField, ls_upflows):

        self.mesh = mesh 

        self.cap = cap
        self.fault = fault
        self.grf_ext = grf_ext 
        self.grf_flt = grf_flt
        self.grf_cap = grf_cap
        self.grf_upflow = grf_upflow

        self.compute_upflow_weightings(ls_upflows)

    def compute_upflow_weightings(self, lengthscale):

        mesh_centre = self.mesh.m.centre[:2]
        col_centres = np.array([c.centre for c in self.mesh.m.column])
        
        dxy = col_centres - mesh_centre
        ds = np.sum(-((dxy / lengthscale) ** 2), axis=1)

        upflow_weightings = np.exp(ds)
        self.upflow_weightings = upflow_weightings

    def sample(self):

        lnperms_ext = self.grf_ext.sample()
        lnperms_cap = self.grf_cap.sample()
        lnperms_flt = self.grf_flt.sample()

        cap_cells = self.cap.sample()

        fault_cells, fault_cols = self.fault.sample()

        lnperms = np.copy(lnperms_ext)
        lnperms[fault_cells] = lnperms_flt[fault_cells]
        lnperms[cap_cells] = lnperms_cap[cap_cells]
        
        upflow_samples = self.grf_upflow.sample() * self.upflow_weightings
        upflows = np.zeros(self.mesh.m.num_columns)
        upflows[fault_cols] = upflow_samples[fault_cols]
        
        return np.concatenate((lnperms, upflows))
    
    def split(self, p_i):

        logks = p_i[:self.mesh.m.num_cells]
        mass_rates_t = p_i[-self.mesh.m.num_columns:]

        upflows = []
        for rate, col in zip(mass_rates_t, self.mesh.m.column):
            if rate > 0:
                upflow = MassUpflow(col.cell[-1], rate)
                upflows.append(upflow)
        
        return logks, upflows
