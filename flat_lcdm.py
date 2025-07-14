import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
# model
from models.flat_lcdm import *

# Read data
df1 = pd.read_csv("data/table_1.csv", sep=r'\s+', comment="#", header=None)
df2 = pd.read_csv("data/table_2.csv", sep=r'\s+', comment="#", header=None)
df = pd.concat([df1, df2], ignore_index=True)

z_data, Hz_data, sigma_data = df[1].values, df[2].values, df[3].values

def run_mcmc(z, Hz, sigma, n_steps=50000, init_params=(70.0, 0.3), step_sizes=(0.1, 0.05), burn_in=5000):
    """
    Custom MCMC sampler for H0 and Om (flat universe assumed, Ok=0).
    """
    ndim = 2
    samples = np.zeros((n_steps, ndim))
    samples[0] = init_params

    # Initial likelihood (fix Ok = 0)
    current_log_likelihood = log_likelihood(samples[0], z, Hz, sigma)

    for i in range(1, n_steps):
        # Propose new parameters
        proposal = np.random.normal(samples[i - 1], step_sizes)

        H0_prop, Om_prop = proposal

        # Flat priors
        if not (50 < H0_prop < 90 and 0.1 < Om_prop < 0.5):
            samples[i] = samples[i - 1]
            continue

        proposed_log_likelihood = log_likelihood((H0_prop, Om_prop), z, Hz, sigma)

        # Acceptance probability
        acceptance_probability = np.exp(proposed_log_likelihood - current_log_likelihood)
        if np.random.rand() < acceptance_probability:
            samples[i] = proposal
            current_log_likelihood = proposed_log_likelihood
        else:
            samples[i] = samples[i - 1]

    return samples[burn_in:]


# Run MCMC
samples = run_mcmc(z_data, Hz_data, sigma_data)

# Extract samples
H0_sample = samples[:, 0]
Om_sample = samples[:, 1]

# Parameter estimates
H0_mean, H0_std = np.mean(H0_sample), np.std(H0_sample)
Om_mean, Om_std = np.mean(Om_sample), np.std(Om_sample)

print(f"H0 = {H0_mean:.2f} ± {H0_std:.2f}")
print(f"Omega_m = {Om_mean:.3f} ± {Om_std:.3f}")

# Trace plot
param_names = ["$H_0$", r"$\Omega_m$"]
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for i in range(2):
    axes[i].plot(samples[:, i], color="black", alpha=0.7, linewidth=0.5)
    axes[i].set_ylabel(param_names[i], fontsize=12)
    axes[i].grid(True)

axes[-1].set_xlabel("Step", fontsize=12)
plt.tight_layout()
fig.savefig("./result/trace_plot_2d.png", dpi=300, bbox_inches="tight")
plt.show()

# corner plot
labels = ["$H_0$", r"$\Omega_m$"]
truths = [67.4, 0.315] # to compare with planck2018
fig = corner.corner(
    samples,
    labels=labels,
    truths=truths,
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 14},
    quantiles=[0.16, 0.5, 0.84],
    title_loc="left"
)
fig.savefig("./result/corner_plot_2d.png", dpi=300, bbox_inches="tight")
plt.show()

