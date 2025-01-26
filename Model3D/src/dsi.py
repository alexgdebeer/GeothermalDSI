import numpy as np
from scipy.stats import norm


class DSIMapping():

    def __init__(self, FGs: np.ndarray):
        """xs should be a matrix that contains all samples of the vector 
        of interest. Each column should be a sample.
        """

        self.dim_x, self.n_samples = FGs.shape
        self.FGs = FGs
        self.FGs_sorted = np.sort(np.copy(FGs))

        self.norm = norm()
        
        self.cdf_elem_size = 1.0 / (self.n_samples - 1)
        self.row_inds = np.arange(self.dim_x, dtype=np.int32)

        # Back-transform samples from the target distribution
        ns = np.array([self._transform_inv(FG_i) for FG_i in self.FGs.T]).T
        
        self.C = np.cov(ns)
        self.C += 1e-8 * np.diag(np.diag(self.C))
        self.L = np.linalg.cholesky(self.C)
        self.L_inv = np.linalg.inv(self.L)
        return

    def _transform_inv(self, FG_i: np.ndarray):
        
        inds_1 = ((FG_i[:, None] - self.FGs_sorted) > -1e-8).sum(axis=1)
        inds_1[inds_1 == self.n_samples] = self.n_samples - 1
        inds_0 = inds_1 - 1

        FGs_0 = self.FGs_sorted[self.row_inds, inds_0]
        FGs_1 = self.FGs_sorted[self.row_inds, inds_1]

        dzs = (FG_i-FGs_0) / (FGs_1-FGs_0)
        zs = self.cdf_elem_size * (inds_0 + dzs)

        # Avoid numerical issues at edges of CDF
        zs = np.clip(zs, 1e-4, 1.0-1e-4)

        # Transform back to unit normal
        us = self.norm.ppf(zs)
        return us
    
    def _transform(self, ns_i: np.ndarray):

        # Correlate and evaluate CDF of unit normal
        zs_i = self.norm.cdf(self.L @ ns_i)
        zs_i = np.clip(zs_i, 1e-4, 1.0-1e-4)

        # Compute inverse CDF
        inds_0 = np.int32(np.floor(zs_i / self.cdf_elem_size))
        zs_loc = (zs_i - inds_0 * self.cdf_elem_size) / self.cdf_elem_size

        FGs_0 = self.FGs_sorted[self.row_inds, inds_0]
        FGs_1 = self.FGs_sorted[self.row_inds, inds_0+1]
        FGs_i = FGs_0 * (1.0 - zs_loc) + FGs_1 * zs_loc
        return FGs_i
    
    def transform_inv(self, FGs: np.ndarray):
        """Transforms a set of variates from the target density back to
        a set of standard normal variates.
        """
        ns = np.array([self._transform_inv(FG_i) for FG_i in FGs.T]).T
        return self.L_inv @ ns

    def transform(self, ns: np.ndarray):
        return np.array([self._transform(n) for n in ns.T]).T
    
    def generate_prior_samples(self, ny: int, n: int):
        """Generates a set of samples from the joint distribution of 
        the data and predictive QoIs.
        """

        samples_pri_n = np.random.normal(size=(self.dim_x, n))
        samples_pri = self.transform(samples_pri_n)
        return samples_pri[ny:, :]

    def generate_conditional_samples(self, y_obs: np.ndarray, n: int):

        # Compute inverse of y_obs
        ny = len(y_obs)
        y_obs_0 = np.hstack((y_obs, np.zeros(self.dim_x-ny)))[:, None]
        n_obs = self.transform_inv(y_obs_0)[:ny]
        
        # Check transformations work...
        # n_obs_0 = np.hstack((n_obs.flatten(), np.zeros(self.dim_x-ny)))[:, None]
        # y_obs_1 = self.transform(n_obs_0).flatten()
        # assert np.max(np.abs(y_obs[:ny]-y_obs_1[:ny])) < 1e-4, "Issue with transformations."

        samples_cond_n = np.random.normal(size=(self.dim_x, n))
        samples_cond_n[:ny, :] = n_obs
        samples_cond = self.transform(samples_cond_n)

        return samples_cond[ny:, :]