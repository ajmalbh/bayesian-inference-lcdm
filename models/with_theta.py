import numpy as np
from scipy.integrate import quad

#Constants.
c = 299792.458
z_dec = 1090  

# ΛCDM Model: Flat
def H_model(z, H0, Om, Ok):
    return H0 * np.sqrt(Om * (1 + z)**3 + Ok * (1 +z)**2 + (1 - Om - Ok))

def comoving_distance(H0, Om, Ok): #r  
    d_H = c/H0
    integrand = lambda z: 1/np.sqrt(Om*(1+z)**3 + Ok*(1+z)**2 + 1 - Om - Ok)
    integral, _ = quad(integrand, 0, z_dec)
    return d_H*integral

def angular_diameter_distance(r, Ok, H0): #d_A

    d_H = c/H0
    if Ok == 0:
        return r

    x = np.sqrt(np.abs(Ok)) / d_H
    if Ok < 0:
        return (np.sin(x*r)/x)
    if Ok > 0:
        return (np.sinh(x*r)/x)

def theta_A(H0, Om, Ok):
    r_s = 144.43 #Mpc
    r = comoving_distance(H0, Om, Ok)
    d_A = angular_diameter_distance(r, Ok, H0)
    result = r_s/d_A
    return result


# Log-Likelihood Function
def log_likelihood(theta, z, Hz, sigma):
    theta_star_obs = 1.04108e-2
    sigma_theta_star = 0.0005e-2
    H0, Om, Ok = theta
    model = H_model(z, H0, Om, Ok)
    chi2_Hz = np.sum(((Hz - model) / sigma)**2)
    chi2_theta = ((theta_A(H0, Om, Ok) - theta_star_obs)/sigma_theta_star)**2
    return -0.5 * (chi2_Hz + chi2_theta)


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