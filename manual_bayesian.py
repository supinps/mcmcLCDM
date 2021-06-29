import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from loglikelihood import Running

np.set_printoptions(suppress=True)

if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('figures'):
    os.makedirs('figures')

samples = 20000
burn_in = 1000

# Parameter Initial Values
H0_ini = 68
Om_ini = 0.3
nu_ini = .005

# Parameters of proposal distribution
wH0 = 5
wOm = 0.1
wnu = 0.002
cov_matrix = np.zeros(shape=(3, 3))
cov_matrix[0, 0] = wH0 ** 2
cov_matrix[1, 1] = wOm ** 2
cov_matrix[2, 2] = wnu ** 2

# Seed for random number generation while sampling
seed1 = 1012345
np.random.seed(seed=seed1)

# acc - accepted parameter values
H0_acc = H0_ini
Om_acc = Om_ini
nu_acc = nu_ini

total_samples = np.ndarray(shape=(samples, 3))
accepted_samples = np.ndarray(shape=(samples, 3))
lnL = np.ndarray(shape=(samples, 1))

data_file = "SCPUnion2_mu_vs_z.txt"
raw_data = np.genfromtxt(data_file, delimiter='\t', skip_header=5, names=True)
SNe307 = Running(raw_data)
lnL[0] = SNe307.log_likelihood([H0_acc, Om_acc, nu_acc])

# Following loop does the Markov Chain Monte Carlo (MCMC) sampling of the distribution.
n_accept = 0
for i in range(1, samples, 1):
    # gaussian proposal distribution, zero correlation between parameters.
    H0_temp, Om_temp, nu_temp = np.random.multivariate_normal(mean=np.asarray([H0_acc, Om_acc, nu_acc]), cov=cov_matrix, size=None)
    params_temp = np.asarray([H0_temp, Om_temp, nu_temp])
    total_samples[i, :] = params_temp

    lnL[i] = SNe307.log_likelihood([H0_temp, Om_temp, nu_temp])

    # Metropolis rule
    if lnL[i] > lnL[i - 1]:  # accept proposed point
        H0_acc = H0_temp
        Om_acc = Om_temp
        nu_acc = nu_temp
        accepted_samples[i, :] = np.asarray([H0_acc, Om_acc, nu_acc])
        n_accept += 1
    else:
        alpha = np.random.random_sample(size=None)
        if lnL[i] - lnL[i - 1] > np.log(alpha):  # accept proposed point
            H0_acc = H0_temp
            Om_acc = Om_temp
            nu_acc = nu_temp
            accepted_samples[i, :] = np.asarray([H0_acc, Om_acc, nu_acc])
            n_accept += 1
        else:  # reject proposed point
            # chain stays at the currant point.
            # Currant (not proposed) point is read to the accepted sample.
            accepted_samples[i, :] = np.asarray([H0_acc, Om_acc, nu_acc])
            lnL[i] = lnL[i - 1]
        print("\rAcceptance Ratio {0}: {1}".format(i, n_accept / (1.0 * i)), sep='', end='', flush=True)

# Compute and print the summary of sampled distribution.
H0_mean = np.mean(accepted_samples[burn_in:, 0])
Om_mean = np.mean(accepted_samples[burn_in:, 1])
nu_mean = np.mean(accepted_samples[burn_in:, 2])
H0_std = np.sqrt(np.var(accepted_samples[burn_in:, 0]))
Om_std = np.sqrt(np.var(accepted_samples[burn_in:, 1]))
nu_std = np.sqrt(np.var(accepted_samples[burn_in:, 2]))

#  print("covariance:\n", np.cov(accepted_samples[burn_in:, 1:6], y=None, rowvar=0, bias=0, ddof=None))
print("\nFinal LogLikelihood : ", lnL[-1])
print("H0 =", H0_mean, "+/-", H0_std)
print("Om =", Om_mean, "+/-", Om_std)
print("nu =", nu_mean, "+/-", nu_std)

np.savetxt("output/log_likelihood.txt", lnL, header="logl", delimiter=",")
np.savetxt("output/total_proposed_sample.txt", total_samples, header="$H_0$,$\\Omega_{m}$,$\\nu$", delimiter=",", comments='')
np.savetxt("output/accepted_sample.txt", accepted_samples, header="$H_0$,$\\Omega_{m}$,$\\nu$", delimiter=",", comments='')

df = pd.DataFrame(lnL)
df.plot()
plt.xlabel("Iterations")
plt.ylabel("Log-Likelihood")
plt.show()
