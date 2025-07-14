import numpy as np

# ΛCDM Model: Flat
def H_model(z, H0, Om, Ok):
    return H0 * np.sqrt(Om * (1 + z)**3 + Ok * (1 +z)**2 + (1 - Om - Ok))

# Log-Likelihood Function
def log_likelihood(theta, z, Hz, sigma):
    H0, Om, Ok = theta
    model = H_model(z, H0, Om, Ok)
    chi2_Hz = np.sum(((Hz - model) / sigma)**2)
    return -0.5 * (chi2_Hz)


# Priors: Flat for H0, Om, and Ok
def log_prior(theta):
    H0, Om, Ok = theta
    if 50 < H0 < 90 and 0.1 < Om < 0.5 and -0.5 < Ok < 0.5:
        return 0.0  # flat prior
    return -np.inf  # log(0)

# Posterior Probability
def log_posterior(theta, z, Hz, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, Hz, sigma)