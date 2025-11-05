import matplotlib

matplotlib.use("TkAgg")  # Use Tk backend instead of default (avoids OpenGL issues)
import matplotlib.pyplot as plt
import numpy as np


def noise_psd(frequencies, noise_level, knee_frequency, alpha):
    """
    Calculate the power spectral density (PSD) of noise.

    Parameters:
    frequencies (np.ndarray): Array of frequency values.
    noise_level (float): The white noise level.
    knee_frequency (float): The knee frequency where the PSD transitions.
    alpha (float): The spectral index.

    Returns:
    np.ndarray: The calculated PSD values.
    """
    psd = noise_level**2 * (1 + (np.abs(frequencies) / knee_frequency) ** alpha)
    return psd


np.random.seed(123)

# Data parameters
N = 10_000  # Number of samples
dt = 0.02  # time step in seconds
alpha = -1.04  # spectral index
knee_frequency = 8  # Hz
sigma_wn = 1

frequencies = np.fft.rfftfreq(N, dt)

# Noise model
psd_model = noise_psd(
    frequencies,
    noise_level=sigma_wn,
    knee_frequency=knee_frequency,
    alpha=alpha,
)
# Setting first mode of model to 1 to inherit white noise mean level
psd_model[0] = 1


# Sampling white noise in real space
n_wn = np.random.normal(5, 14, size=N)

fft_wn = np.fft.rfft(n_wn)

# Convolving white noise with the noise model in Fourier space
n_corr = np.fft.irfft(np.sqrt(psd_model) * fft_wn)

# fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(
    np.arange(N) * dt,
    n_wn,
    label="White noise",
    color="k",
    alpha=0.5,
)

plt.plot(
    np.arange(N) * dt,
    n_corr,
    label="Correlated noise",
    color="r",
    alpha=0.5,
)

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Noise amplitude")
plt.title("Noise Comparison")

plt.show()
