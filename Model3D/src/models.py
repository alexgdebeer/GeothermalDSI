from copy import deepcopy
from enum import Enum
from itertools import product
import os

import layermesh.mesh as lm
import numpy as np
import pyvista as pv
import pywaiwera
from scipy.spatial import Delaunay
import yaml

from src import utils
from src.consts import *


EPS = 1e-8


class ExitFlag(Enum):
    SUCCESS = 1
    FAILURE = 2


class Mesh():

    def __init__(self, name):

        self.name = name
        self.m: lm.mesh = lm.mesh(f"{self.name}.h5")

        self.cell_centres = [c.centre for c in self.m.cell]
        self.col_centres = [c.centre for c in self.m.column]
        self.tri = Delaunay(self.cell_centres)
        
        self.col_cells = {
            col.index: [c.index for c in col.cell] 
            for col in self.m.column}
        
        self.fem_mesh = pv.UnstructuredGrid(f"{self.name}.vtu")

        centers = self.fem_mesh.cell_centers().points
        self.mesh_mapping = np.array([self.m.find(p, indices=True) 
                                      for p in centers])


class MassUpflow():
    def __init__(self, cell: lm.cell, rate: float):
        self.cell = cell
        self.rate = rate


class Well():
    
    def __init__(self, x: float, y: float, depth: float, mesh, 
                 feedzone_depth: float, feedzone_rate: float):
        
        self.x = x
        self.y = y

        self.max_elev = mesh.m.find((x, y, depth)).column.surface
        self.min_elev = depth
        self.coords = np.array([[x, y, self.max_elev], 
                                [x, y, self.min_elev]])

        self.feedzone_cell = mesh.m.find((x, y, feedzone_depth))
        self.feedzone_rate = feedzone_rate


class PermField():

    def __init__(self, std, grf, level_func_perm, level_func_por):

        self.std = std
        self.grf = grf
        self.level_func_perm = level_func_perm
        self.level_func_por = level_func_por
        self.n_params = len(self.grf.points)

    def level_set_perm(self, ps):
        return np.array([self.level_func_perm(p) for p in ps])
    
    def level_set_por(self, ps):
        return np.array([self.level_func_por(p) for p in ps])
    
    def sample(self):
        """Generates a random sample of permeabilities and porosities."""

        ws = np.random.normal(size=self.n_params)
        ps = self.std * self.grf.transform(ws)

        perms = self.level_set_perm(ps)
        pors = self.level_set_por(ps)
        return perms, pors


class UpflowField():

    def __init__(self, mu, std, grf):

        self.mu = mu
        self.std = std
        self.grf = grf

        self.n_params = self.grf.points.shape[0]

    def sample(self):
        """Generates a random set of upflows."""

        ws = np.random.normal(size=self.n_params)
        return self.mu + self.std * self.grf.transform(ws)


class Fault():

    def __init__(self, mesh: Mesh, bounds):
        
        self.mesh = mesh
        self.x0 = self.mesh.m.bounds[0][0]
        self.x1 = self.mesh.m.bounds[1][0]
        self.bounds = bounds
    
    def sample(self):
        """Samples a set of fault cells / columns."""

        y0 = np.random.uniform(*self.bounds[0])
        y1 = np.random.uniform(*self.bounds[1])

        cols = self.mesh.m.column_track([[self.x0, y0], [self.x1, y1]])
        cols = [c[0] for c in cols]

        cells = [cell for col in cols for cell in col.cell]

        col_inds = [col.index for col in cols]
        cell_inds = [cell.index for cell in cells]

        return cell_inds, col_inds


class ClayCap():

    def __init__(self, mesh: Mesh, bounds, n_terms, coef_sds):
        
        self.cell_centres = np.array([c.centre for c in mesh.m.cell])
        self.col_centres = np.array([c.centre for c in mesh.m.column])

        self.cx = mesh.m.centre[0]
        self.cy = mesh.m.centre[1]

        self.bounds = bounds 
        self.n_terms = n_terms 
        self.coef_sds = coef_sds
        self.coefs_shape = (self.n_terms, self.n_terms, 4)
        self.n_coefs = 4 * self.n_terms ** 2

    def cartesian_to_spherical(self, ds):

        rs = np.linalg.norm(ds, axis=1)
        phis = np.arccos(ds[:, 2] / rs)
        thetas = np.arctan2(ds[:, 1], ds[:, 0])
        
        return rs, phis, thetas
    
    def compute_cap_radii(self, phis, thetas, width_h, width_v, coefs):
        """Computes the radius of the clay cap in the direction of each 
        cell, by taking the radius of the (deformed) ellipse that forms 
        the base of the cap, then adding the randomised Fourier series 
        to it.
        """

        rs = np.sqrt(((np.sin(phis) * np.cos(thetas) / width_h) ** 2
                       + (np.sin(phis) * np.sin(thetas) / width_h) ** 2
                       + (np.cos(phis) / width_v)**2) ** -1)
        
        for n, m in product(range(self.n_terms), range(self.n_terms)):
        
            rs += (coefs[n, m, 0] * np.cos(n * thetas) * np.cos(m * phis)
                   + coefs[n, m, 1] * np.cos(n * thetas) * np.sin(m * phis)
                   + coefs[n, m, 2] * np.sin(n * thetas) * np.cos(m * phis)
                   + coefs[n, m, 3] * np.sin(n * thetas) * np.sin(m * phis))
        
        return rs

    def sample(self):
        """Returns an array of booleans that indicate whether each cell 
        is contained within the clay cap.
        """

        cz = np.random.uniform(*self.bounds[0])
        width_h = np.random.uniform(*self.bounds[1])
        width_v = np.random.uniform(*self.bounds[2])
        dip = np.random.uniform(*self.bounds[3])

        coefs = np.random.uniform(size=self.n_coefs)
        coefs = np.reshape(self.coef_sds * coefs, self.coefs_shape)

        centre = np.array([self.cx, self.cy, cz])
        ds = self.cell_centres - centre
        ds[:, -1] += (dip / width_h**2) * (ds[:, 0]**2 + ds[:, 1]**2) 

        cell_radii, cell_phis, cell_thetas = self.cartesian_to_spherical(ds)

        cap_radii = self.compute_cap_radii(cell_phis, cell_thetas,
                                           width_h, width_v, coefs)

        return np.where(cell_radii < cap_radii)


class Model():
    """Base class for models, with a set of default methods."""

    def __init__(
        self, 
        path: str, 
        mesh: Mesh, 
        perms: np.ndarray, 
        pors: np.ndarray,
        wells: list, 
        upflows: list, 
        dt: float, 
        tmax: float
    ):

        self.ns_path = f"{path}_NS"
        self.pr_path = f"{path}_PR"
        self.incon_path = f"{path}_incon"

        self.mesh = mesh 
        self.pors = pors
        self.perms = perms 
        self.wells = wells
        self.upflows = upflows

        self.dt = dt
        self.tmax = tmax
        self.ts = np.arange(0, tmax, dt)

        self.feedzone_cell_inds = [w.feedzone_cell.index for w in wells]
        self.n_feedzones = len(self.feedzone_cell_inds)

        self.ns_model = None
        self.pr_model = None
        
        self.generate_ns()
        self.generate_pr()

    def initialise_ns_model(self):
        
        self.ns_model = {
            "eos": {"name": "we"},
            "gravity": GRAVITY,
            "logfile": {"echo": False},
            "mesh": {
                "filename": f"{self.mesh.name}.msh"
            },
            "title": "3D Model"
        }

    def add_boundaries(self):
        """Adds an atmospheric boundary condition to the top of the 
        model (leaves the sides with no-flow conditions).
        """

        self.ns_model["boundaries"] = [{
            "primary": [P_ATM, T_ATM], 
            "region": 1,
            "faces": {
                "cells": [c.index for c in self.mesh.m.surface_cells],
                "normal": [0, 0, 1]
            }
        }]

    def add_upflows(self):
        """Adds the mass upflows to the base of the model. Where there 
        are no mass upflows, a background heat flux of constant 
        magnitude is imposed.
        """

        upflow_cell_inds = [
            upflow.cell.index 
            for upflow in self.upflows]

        heat_cell_inds = [
            c.cell[-1].index for c in self.mesh.m.column 
            if c.cell[-1].index not in upflow_cell_inds]

        self.ns_model["source"] = [{
            "component": "energy",
            "rate": HEAT_RATE * self.mesh.m.cell[c].column.area,
            "cell": c
        } for c in heat_cell_inds]

        self.ns_model["source"].extend([{
            "component": "water",
            "enthalpy": MASS_ENTHALPY, 
            "rate": u.rate * u.cell.column.area,
            "cell": u.cell.index
        } for u in self.upflows])

        total_mass = sum([u.rate * u.cell.column.area for u in self.upflows])
        utils.info(f"Total mass input: {total_mass:.2f} kg/s")

    def add_rocktypes(self):
        """Adds rocks with given permeabilities (and constant porosity, 
        conductivity, density and specific heat) to the model. 
        Permeabilities may be isotropic or anisotropic.
        """
        
        if len(self.perms) != self.mesh.m.num_cells:
            raise Exception("Incorrect number of permeabilities.")
        
        self.ns_model["rock"] = {"types": [{
            "name": f"{c.index}",
            "porosity": self.pors[c.index], 
            "permeability": 10.0 ** self.perms[c.index],
            "cells": [c.index],
            "wet_conductivity": CONDUCTIVITY,
            "dry_conductivity": CONDUCTIVITY,
            "density": DENSITY,
            "specific_heat": SPECIFIC_HEAT
        } for c in self.mesh.m.cell]}

    def add_wells(self):
        """Adds wells with constant production / injection rates to the 
        model.
        """

        self.pr_model["source"].extend([{
            "component": "water",
            "rate": w.feedzone_rate,
            "cell": w.feedzone_cell.index,
            "interpolation": "step"
        } for w in self.wells])

    def add_ns_incon(self):
        """Adds path to initial condition file to the model, if the 
        file exists. Otherwise, sets the entire model to a constant 
        temperature and pressure.
        """

        if os.path.isfile(f"{self.incon_path}.h5"):
            self.ns_model["initial"] = {"filename": f"{self.incon_path}.h5"}
        else:
            self.ns_model["initial"] = {"primary": [P0, T0], "region": 1}

    def add_pr_incon(self):
        """Sets the production history initial condition to be the 
        output file from the natural state run.
        """
        
        self.pr_model["initial"] = {"filename": f"{self.ns_path}.h5"}
    
    def add_ns_timestepping(self):
        """Sets the natural state timestepping parameters."""

        self.ns_model["time"] = {
            "step": {
                "size": 1.0e+6,
                "adapt": {"on": True}, 
                "maximum": {"number": MAX_NS_TSTEPS},
                "method": "beuler",
                "stop": {"size": {"maximum": NS_STEPSIZE}}
            }
        }

    def add_pr_timestepping(self):
        """Sets the production history timestepping parameters."""

        self.pr_model["time"] = {
            "step": {
                "adapt": {"on": True},
                "size": self.dt,
                "maximum": {"number": MAX_PR_TSTEPS},
            },
            "stop": self.tmax
        }

    def add_ns_output(self):
        """Sets up the natural state simulation such that it only saves 
        the final model state.
        """

        self.ns_model["output"] = {
            "frequency": 0, 
            "initial": False, 
            "final": True
        }

    def add_pr_output(self):
        """Sets up production history checkpoints."""
        
        self.pr_model["output"] = {
            "checkpoint": {
                "time": [self.dt], 
                "repeat": True
            },
            "frequency": 0,
            "initial": True,
            "final": False
        }

    def generate_ns(self):
        """Generates the natural state model."""
        
        self.initialise_ns_model()
        self.add_boundaries()
        self.add_rocktypes()
        self.add_upflows()
        self.add_ns_incon()
        self.add_ns_timestepping()
        self.add_ns_output()

        utils.save_json(self.ns_model, f"{self.ns_path}.json")

    def generate_pr(self):
        """Generates the production history model."""

        self.pr_model = deepcopy(self.ns_model)

        self.add_wells()
        self.add_pr_timestepping()
        self.add_pr_output()
        self.add_pr_incon()

        utils.save_json(self.pr_model, f"{self.pr_path}.json")

    def delete_output_files(self):
        """Removes output files from previous simulation."""

        for fname in [f"{self.ns_path}.h5", f"{self.pr_path}.h5"]:
            try:
                os.remove(fname)
            except OSError:
                pass

    @utils.timer
    def run(self):
        """Simulates the model and returns a flag that indicates 
        whether the simulation was successful.
        """

        self.delete_output_files()

        env = pywaiwera.docker.DockerEnv(check=False, verbose=False)
        env.run_waiwera(f"{self.ns_path}.json", noupdate=True)
        
        flag = self.get_exitflag(self.ns_path)
        if flag == ExitFlag.FAILURE: 
            return flag

        env = pywaiwera.docker.DockerEnv(check=False, verbose=False)
        env.run_waiwera(f"{self.pr_path}.json", noupdate=True)
        
        return self.get_exitflag(self.pr_path)

    def get_exitflag(self, log_path):
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