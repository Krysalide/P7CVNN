import torch

def complex_mse_per_antenna(pred, target):
    # Erreur complexe point à point
    error = torch.abs(pred - target) ** 2  # Shape: [B, A, H, W]
    # Moyenne sur batch et spatial
    return error.mean(dim=(0, 2, 3))  # Shape: [A]

def complex_mae_per_antenna(pred, target):
    error = torch.abs(pred - target)
    return error.mean(dim=(0, 2, 3))  # Shape: [A]

def phase_error_per_antenna(pred, target):
    # angle en radians ∈ [−π, π]
    phase_diff = torch.angle(pred) - torch.angle(target)
    # Ramène l'erreur entre -π et π
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    return torch.abs(phase_diff).mean(dim=(0, 2, 3))  # Shape: [A]


def magnitude_error_per_antenna(pred, target):
    
    magnitude_diff = torch.abs(torch.abs(pred) - torch.abs(target))
    
    return magnitude_diff.mean(dim=(0, 2, 3))


def relative_error_per_antenna(pred, target, eps=1e-8):
    rel_error = torch.abs(pred - target) / (torch.abs(target) + eps)
    return rel_error.mean(dim=(0, 2, 3))  # Shape: [A]


def real_imag_mse_per_antenna(pred, target):
    real_mse = ((pred.real - target.real) ** 2).mean(dim=(0, 2, 3))
    imag_mse = ((pred.imag - target.imag) ** 2).mean(dim=(0, 2, 3))
    return real_mse, imag_mse  # Shape: ([A], [A])

if __name__ == '__main__':
    pred = torch.randn(1, 16, 256, 512, dtype=torch.cfloat)
    target = pred + 0.1 * torch.randn(1, 16, 256, 512, dtype=torch.cfloat)

    mse = complex_mse_per_antenna(pred, target)
    mae = complex_mae_per_antenna(pred, target)
    phase = phase_error_per_antenna(pred, target)
    mag = magnitude_error_per_antenna(pred, target)
    rel = relative_error_per_antenna(pred, target)
    real_mse, imag_mse = real_imag_mse_per_antenna(pred, target)

    print("MSE par antenne:", mse)
    print('MSE shape:', mse.shape)
    print("MAE par antenne:", mae)
    print('MAE shape:', mae.shape)
    print("Erreur de phase (rad) par antenne:", phase)
    print("Erreur de phase (deg) par antenne:", phase * 180 / 3.14159)
    print("Erreur sur l'amplitude  par antenne:",mag)
    print('Phase shape:', phase.shape)
    print("Erreur relative par antenne:", rel)
    print('Erreur relative shape:', rel.shape)
    print("MSE réelle par antenne:", real_mse)
    print("MSE réelle shape:", real_mse.shape)
    print("MSE imaginaire par antenne:", imag_mse)
    print("MSE imaginaire shape:", imag_mse.shape)