import jax
import jax.numpy as jnp

@jax.jit
def gaussian_bandpass_filter(signal,freq, fs, fwhm=5.0):
    """
    Apply a Gaussian bandpass filter to a signal in the frequency domain.

    This function creates a Gaussian filter centered at the specified frequency
    (and its negative counterpart) with a given full-width half maximum (FWHM)
    and applies it to the signal in the frequency domain.
    """


    # Convert FWHM to sigma for the Gaussian (in Hz)
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))

    # Create frequency-domain Gaussian filter
    num_data = len(signal)
    freqs = jnp.fft.fftfreq(num_data, d=1/fs)
    gaussian_filter = (jnp.exp(-0.5 * ((freqs - freq) / sigma)**2) +
                       jnp.exp(-0.5 * ((freqs + freq) / sigma)**2))

    # Apply filter in the frequency domain
    data_fft = jnp.fft.fft(signal)
    filtered_fft = data_fft * gaussian_filter
    filtered_signal = jnp.fft.ifft(filtered_fft).real

    return filtered_signal