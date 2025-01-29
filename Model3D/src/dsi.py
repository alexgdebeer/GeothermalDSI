import numpy as np
from scipy.stats import norm


Z_MIN = 1e-4


class DSIMapping():

    def __init__(self, GFs: np.ndarray, GFs_min, GFs_max):
        """xs should be a matrix that contains all samples of the vector 
        of interest. Each column should be a sample.
        """

        GFs_aug = np.hstack((GFs, GFs_min[:, None], GFs_max[:, None]))

        self.GFs_sorted = np.sort(GFs_aug)
        self.dim_x, self.n_samples = self.GFs_sorted.shape
        
        if not np.all(self.GFs_sorted[:, 0] == GFs_min):
            raise Exception("Minimum bounds are too strict.")
        if not np.all(self.GFs_sorted[:, -1] == GFs_max):
            raise Exception("Maximum bounds are too strict.")

        self.norm = norm()
        
        self.cdf_elem_size = 1.0 / (self.n_samples - 1)
        self.row_inds = np.arange(self.dim_x, dtype=np.int32)

        # Back-transform samples from the target distribution
        ns = np.array([self._transform_inv(FG_i) for FG_i in GFs.T]).T
        
        self.C = np.cov(ns)
        self.C += 1e-8 * np.diag(np.diag(self.C))
        self.L = np.linalg.cholesky(self.C)
        self.L_inv = np.linalg.inv(self.L)
        return

    def _transform_inv(self, FG_i: np.ndarray):
        
        inds_1 = ((FG_i[:, None] - self.GFs_sorted) > -1e-8).sum(axis=1)
        inds_1[inds_1 == self.n_samples] = self.n_samples - 1
        inds_0 = inds_1 - 1

        GFs_0 = self.GFs_sorted[self.row_inds, inds_0]
        GFs_1 = self.GFs_sorted[self.row_inds, inds_1]

        dzs = (FG_i-GFs_0) / (GFs_1-GFs_0)
        zs = self.cdf_elem_size * (inds_0 + dzs)

        mask_l = zs < self.cdf_elem_size
        mask_r = zs > 1 - self.cdf_elem_size
        mask_c = np.bitwise_and(~mask_l, ~mask_r)

        # Rescale stuff to account for less probability at the edges
        zs[mask_l] *= (Z_MIN / self.cdf_elem_size)
        zs[mask_r] = 1.0 - Z_MIN * (1 - zs[mask_r]) / self.cdf_elem_size
        zs[mask_c] = Z_MIN + (zs[mask_c] - self.cdf_elem_size) * (1 - 2*Z_MIN) / (1 - 2*self.cdf_elem_size)

        # Avoid numerical issues at edges of CDF
        zs = np.clip(zs, 1e-8, 1.0-1e-8)

        # Transform back to unit normal
        us = self.norm.ppf(zs)
        return us
    
    def _transform(self, ns_i: np.ndarray):

        # Correlate and evaluate CDF of unit normal
        zs_i = self.norm.cdf(self.L @ ns_i)

        zs_i = np.clip(zs_i, 1e-8, 1.0-1e-8)

        mask_l = zs_i < Z_MIN
        mask_r = zs_i > 1 - Z_MIN
        mask_c = np.bitwise_and(~mask_l, ~mask_r)

        zs_i[mask_l] *= (self.cdf_elem_size / Z_MIN)
        zs_i[mask_r] = 1.0 - self.cdf_elem_size * (1 - zs_i[mask_r]) / Z_MIN 
        zs_i[mask_c] = self.cdf_elem_size + (zs_i[mask_c] - Z_MIN) * (1 - 2*self.cdf_elem_size) / (1 - 2*Z_MIN)

        # Compute inverse CDF
        inds_0 = np.int32(np.floor(zs_i / self.cdf_elem_size))
        zs_loc = (zs_i - inds_0 * self.cdf_elem_size) / self.cdf_elem_size

        GFs_0 = self.GFs_sorted[self.row_inds, inds_0]
        GFs_1 = self.GFs_sorted[self.row_inds, inds_0+1]
        GFs_i = GFs_0 * (1.0 - zs_loc) + GFs_1 * zs_loc
        return GFs_i
    
    def transform_inv(self, GFs: np.ndarray):
        """Transforms a set of variates from the target density back to
        a set of standard normal variates.
        """
        ns = np.array([self._transform_inv(FG_i) for FG_i in GFs.T]).T
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
        n_obs_0 = np.hstack((n_obs.flatten(), np.zeros(self.dim_x-ny)))[:, None]
        y_obs_1 = self.transform(n_obs_0).flatten()
        assert np.max(np.abs(y_obs[:ny]-y_obs_1[:ny])) < 1e-4, "Issue with transformations."

        samples_cond_n = np.random.normal(size=(self.dim_x, n))
        samples_cond_n[:ny, :] = n_obs
        samples_cond = self.transform(samples_cond_n)

        return samples_cond[ny:, :]