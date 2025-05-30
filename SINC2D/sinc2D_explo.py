import numpy as np
import matplotlib.pyplot as plt

def h_K_fs(k, fl, b, fs, K):
    """
    Filtre sinc 1D passe-bande :
    h_K_fs(k, fl, b) = (2(fl + b)/fs) * sinc(2(fl + b)(k - K//2)) - (2fl/fs) * sinc(2fl(k - K//2))
    """
    k = np.array(k)
    k0 = K // 2

    term1 = (2 * (fl + b) / fs) * np.sinc(2 * (fl + b) * (k - k0))
    term2 = (2 * fl / fs) * np.sinc(2 * fl * (k - k0))
    return term1 - term2

def cosine_weight_2D(N, M):
    """
    Fonction de pondération cosinus en 2D (type Hanning 2D).
    """
    w_n = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    w_m = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(M) / (M - 1))
    return np.outer(w_n, w_m)

def sinc_2D(N, M, fs_st, fl_st, b_st, fs_ft, fl_ft, b_ft):
    """
    Génère un filtre 2D basé sur le produit de deux filtres sinc 1D pondérés.
    
    Paramètres :
    - N, M : dimensions slow-time et fast-time
    - fs_st, fl_st, b_st : sampling, cutoff, bande pour slow-time
    - fs_ft, fl_ft, b_ft : sampling, cutoff, bande pour fast-time
    """
    n = np.arange(N)
    m = np.arange(M)

    # Filtres sinc 1D selon ta définition
    h_n = h_K_fs(n, fl_st, b_st, fs_st, N)
    h_m = h_K_fs(m, fl_ft, b_ft, fs_ft, M)

    # Produit extérieur → filtre 2D brut
    h_2d = np.outer(h_n, h_m)

    # Fenêtre de pondération 2D
    w_2d = cosine_weight_2D(N, M)

    # Filtre final
    return w_2d * h_2d


# Paramètres du filtre
N, M = 51, 51
fs_st, fl_st, b_st = 1000, 100, 50
fs_ft, fl_ft, b_ft = 1000, 200, 100

# Calcul du filtre
filt2D = sinc_2D(N, M, fs_st, fl_st, b_st, fs_ft, fl_ft, b_ft)

# Affichage
plt.figure(figsize=(6, 5))
plt.imshow(filt2D, cmap='seismic', extent=[0, M, 0, N])
plt.colorbar(label='Amplitude')
plt.title('Filtre Sinc 2D pondéré (slow-time vs fast-time)')
plt.xlabel('Fast-time (m)')
plt.ylabel('Slow-time (n)')
plt.tight_layout()
plt.show()
