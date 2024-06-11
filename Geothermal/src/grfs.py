from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import pyvista as pv

from src import utils


class BC(Enum):
    NEUMANN = 1
    ROBIN = 2


class GRF(ABC):

    def load_matrices(self):
        """Loads the correlation matrix of a GRF, as well as its 
        Cholesky factorisation.
        """

        try:
            self.R = np.load(f"{self.folder}/R.npy")
            self.L = np.load(f"{self.folder}/L.npy")

        except FileNotFoundError:
            utils.info("Precomputed correlation matrix not found. Constructing...")
            self.generate_matrices()
            np.save(f"{self.folder}/R", self.R)
            np.save(f"{self.folder}/L", self.L)

    def generate_matrices(self):
        """Generates the correlation matrix of a GRF, as well as its 
        Cholesky factorisation.
        """

        self.R = self.k.compute_R(self.points)
        utils.info("Correlation matrix generated.")
        self.L = np.linalg.cholesky(self.R)
        return 

    def transform(self, ws):
        """Transforms a set of variates drawn from the unit normal 
        distribution.
        """

        return self.L @ ws


class Kernel(ABC):
    @abstractmethod
    def compute_R():
        pass


class SquaredExp(ABC):
    
    def __init__(self, lengthscale):
        self.lsq = lengthscale ** 2

    def compute_R(self, xs):

        dsq = np.zeros((len(xs), len(xs)))

        for (i, xi) in enumerate(xs):
            for (j, xj) in enumerate(xs):
                dsq[i, j] = np.sum((xi-xj)**2 / self.lsq)

        R = np.exp(-0.5 * dsq) 
        R += 1e-6 * np.eye(len(xs))
        return R


class GRF2D(GRF):

    def __init__(self, mesh, kernel, folder=""):

        self.k = kernel
        self.points = np.array([[*col.centre, 0] for col in mesh.m.column])
        
        self.m = mesh
        self.fem_mesh = pv.PolyData(self.points).delaunay_2d() # needed?

        self.folder = folder
        self.load_matrices()
    
    def plot(self, values, **kwargs):
        """Generates a 3D visualisation of the mesh using PyVista."""
        p = pv.Plotter()
        p.add_mesh(self.fem_mesh, scalars=values, **kwargs)
        p.show()
    
    def layer_plot(self, values, **kwargs):
        """Generates a visualisation of the mesh using Layermesh."""
        col_values = [values[c.column.index] for c in self.m.cell]
        self.m.layer_plot(value=col_values, **kwargs)


class GRF3D(GRF):

    def __init__(self, mesh, kernel, folder=""):

        self.k = kernel
        self.points = np.array([c.centre for c in mesh.m.cell])

        self.mesh = mesh
        self.fem_mesh = mesh.fem_mesh

        self.folder = folder
        self.load_matrices()

    def plot(self, values, **kwargs):
        p = pv.Plotter()
        p.add_mesh(self.fem_mesh, scalars=values[self.mesh.mesh_mapping], **kwargs)
        p.show()

    def slice_plot(self, values, **kwargs):
        self.mesh.m.slice_plot(value=values, **kwargs)