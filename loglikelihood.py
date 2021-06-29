from scipy.integrate import quad
import numpy as np
from numba import jitclass, float64, int64

spec = [('z', float64[:]),
        ('m_data', float64[:]),
        ('sigma', float64[:]),
        ('n', int64)]


@jitclass(spec)
class Running:
    def __init__(self, raw_data):
        data = raw_data[raw_data['z'].argsort()]
        self.z = data['z']
        self.m_data = data['m']
        self.sigma = data['err']
        self.n = data.shape[0]

    def model_fn(self, H0, Om, nu, c=3 * 10 ** 5):
        z = self.z
        mu = []
        for i in range(self.n):
            D_L, _ = quad(
                lambda z1: 1 / (H0 * np.sqrt(1 + (Om / (1 - nu)) * ((1 / (1 + z1)) ** (-3 * (1 - nu)) - 1))),
                0, z[i])
            D_L = (c * (1 + z[i])) * D_L
            mu.append(5 * np.log10(D_L) + 25)
        return mu

    def log_likelihood(self, params):
        if params[2] < 0:
            return -np.infty
        lnL = -0.5 * np.sum(
            ((self.m_data - self.model_fn(H0=params[0], Om=params[1], nu=params[2])) ** 2) / self.sigma ** 2)
        return lnL


class LCDM:
    def __init__(self, data_file):
        raw_data = np.genfromtxt(data_file, delimiter='\t', skip_header=4, names=True)
        self.data = raw_data[raw_data['z'].argsort()]
        self.z = self.data['z']
        self.m_data = self.data['m']
        self.sigma = self.data['err']
        self.n = self.data.shape[0]

    def model_fn(self, H0, Ol, c=3 * 10 ** 5):
        z = self.z
        mu = []
        for i in range(307):
            D_L, _ = quad(lambda z1: 1 / (H0 * np.sqrt((1 - Ol) * (1 + z1) ** 3 + Ol)), 0, z[i])
            D_L = (c * (1 + z[i])) * D_L
            mu.append(5 * np.log10(D_L) + 25)
        return mu

    def log_likelihood(self, params):
        lnL = -0.5 * np.sum(
            ((self.m_data - self.model_fn(H0=params[0], Ol=params[1])) ** 2) / self.sigma ** 2)
        return lnL
