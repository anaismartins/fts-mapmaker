import numpy as np
import matplotlib.pyplot as plt


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
    psd = noise_level**2 * (1 + (frequencies / knee_frequency) ** alpha)
    return psd


# Data
N = 10_00
dt = 0.02  # time step in seconds
data = np.random.normal(0, 1, size=N)

alpha = -1.04
knee_frequency = 0.1  # Hz
sigma_wn = 1

psd_wn = np.abs(np.fft.rfft(data)) ** 2 / (N / 2 + 1)
print(np.median(psd_wn), N, dt)
frequencies = np.fft.rfftfreq(N, dt)


# Noise model
psd_values = noise_psd(
    frequencies,
    noise_level=sigma_wn,
    knee_frequency=knee_frequency,
    alpha=alpha,
)


model_real = np.fft.irfft(np.sqrt(psd_values), n=N)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(frequencies, psd_wn, label="White Noise PSD", color="blue")
ax.loglog(frequencies, psd_values, label="Colored Noise PSD", color="red")
ax.axhline(
    np.median(psd_wn), color="magenta", linestyle="--", label="White Noise Level"
)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density")
ax.legend()


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data)
ax.plot(model_real)
plt.show()
