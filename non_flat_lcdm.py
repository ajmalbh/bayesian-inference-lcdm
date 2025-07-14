import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
# model
from models.with_ttpeaks import *

# Read data
df1 = pd.read_csv("data/table_1.csv", sep=r'\s+', comment="#", header=None)
df2 = pd.read_csv("data/table_2.csv", sep=r'\s+', comment="#", header=None)
df = pd.concat([df1, df2], ignore_index=True)

z_data, Hz_data, sigma_data = df[1].values, df[2].values, df[3].values

def run_mcmc(z, Hz, sigma, n_steps=100000, init_params=(70.0, 0.3, 0.0), step_sizes=(0.5, 0.02, 0.02), burn_in=5000):
    """
    Custom MCMC sampler for H0, Om, Ok.

    Parameters:
    - z, Hz, sigma: data
    - n_steps: total MCMC steps
    - init_params: initial guess (H0, Om, Ok)
    - step_sizes: proposal std devs
    - burn_in: steps to discard
    """
    ndim = 3
    samples = np.zeros((n_steps, ndim))
    samples[0] = init_params

    # Initial likelihood
    current_log_likelihood = log_likelihood(samples[0], z, Hz, sigma)

    for i in range(1, n_steps):
        # Propose new parameters
        proposal = np.random.normal(samples[i - 1], step_sizes)

        # Unpack
        H0_prop, Om_prop, Ok_prop = proposal

        # Flat priors
        if not (50 < H0_prop < 90 and 0.1 < Om_prop < 0.5 and -0.5 < Ok_prop < 0.5):
            samples[i] = samples[i - 1]
            continue

        # Proposed likelihood
        proposed_log_likelihood = log_likelihood(proposal, z, Hz, sigma)

        # Acceptance probability
        accept_prob = np.exp(proposed_log_likelihood - current_log_likelihood)
        if np.random.rand() < accept_prob:
            samples[i] = proposal
            current_log_likelihood = proposed_log_likelihood
        else:
            samples[i] = samples[i - 1]

    # Remove burn-in
    return samples[burn_in:]


# run mcmc
samples = run_mcmc(z_data, Hz_data, sigma_data)

# Statistics
H0_sample = samples[:, 0]
Om_sample = samples[:, 1]
Ok_sample = samples[:, 2]

# Parameter estimates
H0_mean = np.mean(H0_sample)
Om_mean = np.mean(Om_sample)
Ok_mean = np.mean(Ok_sample)

H0_std = np.std(H0_sample)
Om_std = np.std(Om_sample)
Ok_std = np.std(Ok_sample)

print(f"H0 = {H0_mean:.2f} ± {H0_std:.2f}")
print(f"Omega_m = {Om_mean:.3f} ± {Om_std:.3f}")
print(f"Omega_k = {Ok_mean:.3f} ± {Ok_std:.3f}")

param_names = ["H0", r"$\Omega_m$", r"$\Omega_k$"]

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

for i in range(3):
    axes[i].plot(samples[:, i], color="black", alpha=0.7, linewidth=0.5)
    axes[i].set_ylabel(param_names[i], fontsize=12)
    axes[i].grid(True)

axes[-1].set_xlabel("Step", fontsize=12)
plt.tight_layout()
fig.savefig("./result/trace_plot_02.png", dpi=300, bbox_inches="tight")
plt.show()

labels = ["H0", r"$\Omega_m$", r"$\Omega_k$"]

fig = corner.corner(
    samples,
    labels=labels,
    truths=None,
    show_titles=True,
    title_fmt=".2f",
    title_kwargs={"fontsize": 12},
    label_kwargs={"fontsize": 14},
    quantiles=[0.16, 0.5, 0.84],
    title_loc="left"
)

fig.savefig("./result/corner_plot_02.png", dpi=300, bbox_inches="tight")
plt.show()