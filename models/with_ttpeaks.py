import numpy as np
from scipy.integrate import quad

print("Running from tt-peaks")

#Constants.
c = 299792.458
z_dec = 1090  
r_s = 144.43 #Mpc


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


def ltt_m(H0, Om, Ok, m=1): # m can take values 1 and 2
    if m == 1:
        phi_m = 0.265
    elif m == 2:
        phi_m = 0.219

    r = comoving_distance(H0, Om, Ok)
    d_A = angular_diameter_distance(r, Ok, H0)
    return (np.pi/r_s)*d_A*(m - phi_m)


# Log-Likelihood Function
def log_likelihood(theta, z, Hz, sigma):
    l1, sigma_l1 = 220.1, 0.8
    l2, sigma_l2 = 546, 10
    H0, Om, Ok = theta
    model = H_model(z, H0, Om, Ok)
    chi2_Hz = np.sum(((Hz - model) / sigma)**2)
    peak_1 = ((l1-ltt_m(H0, Om, Ok))/sigma_l1)**2
    peak_2 = ((l2-ltt_m(H0, Om, Ok, 2))/sigma_l2)**2
    chi2_TTpeak =  peak_1 + peak_2
    return -0.5 * (chi2_Hz + chi2_TTpeak)


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