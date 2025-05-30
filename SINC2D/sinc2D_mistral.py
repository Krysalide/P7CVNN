import numpy as np

def sinc(x):
    """1D sinc function."""
    return np.where(x == 0, 1, np.sin(np.pi * x) / (np.pi * x))

def h_K_fs(k, fl, b, K, fs):
    """1D sinc filter."""
    term1 = 2 * (fl + b) * sinc(2 * (fl + b) * (k - b * K / 2) / fs)
    term2 = 2 * fl * sinc(2 * fl * (k - b * K / 2) / fs)
    return term1 - term2

def sinc_2D(n, m, fl_st, b_st, fl_ft, b_ft, N, M, fs_st, fs_ft):
    """2D sinc filter."""
    # Create a 2D cosine weighting function
    w = np.outer(np.cos(np.linspace(-np.pi/2, np.pi/2, N)), np.cos(np.linspace(-np.pi/2, np.pi/2, M)))

    # Compute 1D filters for slow-time and fast-time
    h_st = h_K_fs(n, fl_st, b_st, N, fs_st)
    h_ft = h_K_fs(m, fl_ft, b_ft, M, fs_ft)

    # Combine them into a 2D filter
    return w * np.outer(h_st, h_ft)

# Example usage
N, M = 51, 51 # Filter lengths
fs_st, fs_ft = 1000, 1000  # Sampling frequencies
fl_st, b_st = 100, 50 # Slow-time lower cutoff frequency and bandwidth
fl_ft, b_ft = 200, 100  # Fast-time lower cutoff frequency and bandwidth

n = np.arange(N)
m = np.arange(M)

sinc_2d_filter = sinc_2D(n[:, np.newaxis], m[np.newaxis, :], fl_st, b_st, fl_ft, b_ft, N, M, fs_st, fs_ft)


import matplotlib.pyplot as plt
plt.imshow(sinc_2d_filter, cmap='jet')
plt.colorbar(label='Amplitude')
plt.title('Filtre Sinc 2D pondéré')
plt.xlabel('Fast-time (m)')
plt.ylabel('Slow-time (n)')
plt.show()