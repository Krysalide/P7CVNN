import torch
import torch.nn.functional as F

def complex_relative_mse_loss_v1(output, target):
    """
    Relative Complex MSE Loss - Version 1:
    (MSE_re / |target_re|^2) + (MSE_im / |target_im|^2)
    This version calculates relative loss for real and imaginary parts separately.
    It's useful when you care about the relative error in each component individually.
    Adds a small epsilon to avoid division by zero.
    """
    output_re = output.real
    output_im = output.imag
    target_re = target.real
    target_im = target.imag

    # Add a small epsilon to avoid division by zero for targets close to zero
    epsilon = 1e-8

    loss_re = F.mse_loss(output_re, target_re, reduction='sum') / (torch.sum(target_re**2) + epsilon)
    loss_im = F.mse_loss(output_im, target_im, reduction='sum') / (torch.sum(target_im**2) + epsilon)

    return loss_re + loss_im

def complex_relative_mse_loss_v2(output, target):
    """
    Relative Complex MSE Loss - Version 2:
    MSE_complex / |target_complex|^2
    This version calculates the MSE of the complex numbers directly and then normalizes
    by the squared magnitude of the target complex numbers. This is often a more
    intuitive "complex relative" loss, as it considers the overall magnitude.
    Adds a small epsilon to avoid division by zero.
    """
    epsilon = 1e-8

    # Calculate squared error for complex numbers: |output - target|^2
    # This is equivalent to (output_re - target_re)^2 + (output_im - target_im)^2
    complex_error = output - target
    squared_complex_error = (complex_error.real**2 + complex_error.imag**2)

    # Calculate squared magnitude of target: |target|^2 = target_re^2 + target_im^2
    squared_target_magnitude = (target.real**2 + target.imag**2)

    # Sum up the squared errors and squared magnitudes for the batch/tensor
    sum_squared_complex_error = torch.sum(squared_complex_error)
    sum_squared_target_magnitude = torch.sum(squared_target_magnitude)

    # Avoid division by zero
    relative_loss = sum_squared_complex_error / (sum_squared_target_magnitude + epsilon)

    return relative_loss

def complex_relative_mse_loss_v3(output, target):
    """
    Relative Complex MSE Loss - Version 3 (Mean Relative Loss):
    Average of (|output_i - target_i|^2 / |target_i|^2) for each element i
    This version calculates the relative loss for each complex element individually
    and then averages them. This can be more robust to outliers with very small
    target magnitudes if you don't want them to dominate the loss when summed.
    Adds a small epsilon to avoid division by zero.
    """
    epsilon = 1e-8

    complex_error = output - target
    squared_complex_error = (complex_error.real**2 + complex_error.imag**2)
    squared_target_magnitude = (target.real**2 + target.imag**2)

    # Calculate element-wise relative squared error
    elementwise_relative_squared_error = squared_complex_error / (squared_target_magnitude + epsilon)

    # Average the element-wise relative squared errors
    relative_loss = torch.mean(elementwise_relative_squared_error)

    return relative_loss

def complex_relative_mse_phase_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Relative Complex MSE Loss with Phase Error (Mean Relative Loss * Mean Phase Error).

    This function calculates:
    1.  The mean of element-wise relative squared magnitude errors:
        (|output_i - target_i|^2 / |target_i|^2) for each complex element i.
    2.  The mean of element-wise squared phase differences between output and target,
        calculated only for elements where the target's magnitude is non-zero.
        The phase difference is wrapped to (-pi, pi] before squaring.
    3.  The final loss is the product of the magnitude-based relative loss and the
        mean phase error.

    Args:
        output (torch.Tensor): The predicted complex tensor.
        target (torch.Tensor): The ground truth complex tensor.

    Returns:
        torch.Tensor: A scalar tensor representing the combined loss.

    Raises:
        ValueError: If input tensors are not complex.
    """
    epsilon = 1e-8 # Small constant to prevent division by zero in magnitude calculation

    # Ensure inputs are complex tensors
    if not torch.is_complex(output) or not torch.is_complex(target):
        raise ValueError("Input tensors must be complex.")
    
    # --- 1. Calculate Magnitude-based Relative MSE Loss ---

    # Calculate the complex error: (output_i - target_i)
    complex_error = output - target

    # Calculate the squared magnitude of the complex error: |error_i|^2 = real_error^2 + imag_error^2
    squared_complex_error_magnitude = (complex_error.real**2 + complex_error.imag**2)

    # Calculate the squared magnitude of the target: |target_i|^2 = real_target^2 + imag_target^2
    squared_target_magnitude = (target.real**2 + target.imag**2)

    # Calculate element-wise relative squared error for magnitude: |error_i|^2 / |target_i|^2
    # Add epsilon to the denominator to prevent division by zero for target_i close to zero.
    elementwise_relative_squared_magnitude_error = squared_complex_error_magnitude / (squared_target_magnitude + epsilon)

    # Take the mean over all elements to get the overall magnitude relative loss
    magnitude_relative_loss = torch.mean(elementwise_relative_squared_magnitude_error)

    # --- 2. Calculate Phase Error ---

    # Create a mask to identify elements where target magnitude is effectively zero.
    # We use a slightly larger threshold (e.g., epsilon * 10) for numerical robustness.
    non_zero_magnitude_mask = squared_target_magnitude >= (epsilon * 10) 

    # Handle the case where all target magnitudes are effectively zero.
    # In this scenario, phase error is undefined/irrelevant, so we set its multiplier to 1.0.
    if not torch.any(non_zero_magnitude_mask):
        # If no elements have non-zero magnitude, the phase component of the loss
        # should not pull the total loss to zero. A multiplier of 1.0 effectively
        # makes the final loss equal to the magnitude_relative_loss.
        mean_phase_error = torch.tensor(1.0, dtype=output.real.dtype, device=output.device)
    else:
        # Get the angle (phase) of output and target tensors in radians, in range (-pi, pi]
        angle_output = torch.angle(output)
        angle_target = torch.angle(target)

        # Calculate the element-wise raw phase difference
        raw_phase_difference = angle_output - angle_target

        # Wrap the phase difference to the range (-pi, pi] to account for periodicity
        # This ensures that a phase difference of 350 degrees is treated as -10 degrees,
        # which is much smaller when squared.
        wrapped_phase_difference = torch.atan2(torch.sin(raw_phase_difference), torch.cos(raw_phase_difference))

        # Calculate the squared phase error for each element
        elementwise_squared_phase_error = wrapped_phase_difference**2

        # Apply the mask: only consider phase errors for elements where target magnitude is non-zero.
        # This prevents NaN propagation from `torch.angle(0 + 0j)`.
        valid_squared_phase_error = elementwise_squared_phase_error[non_zero_magnitude_mask]

        # Take the mean of the valid squared phase errors
        mean_phase_error = torch.mean(valid_squared_phase_error)

    # --- 3. Combine the losses ---
    # As requested, multiply the initial magnitude loss by the mean phase error.
    final_loss = magnitude_relative_loss * mean_phase_error

    return final_loss

# For test!!!
if __name__ == "__main__":
    #
    output_complex = torch.randn(5, dtype=torch.complex64)
    target_complex = torch.randn(5, dtype=torch.complex64)

    # 
    target_complex[0] = 0.01 + 0.02j
    output_complex[0] = 0.02 + 0.03j
    target_complex[1] = 1e-5 + 1e-5j # Very small target

    print(f"Output: {output_complex}")
    print(f"Target: {target_complex}")

    # Original MSE loss for comparison
    output_re = output_complex.real
    output_im = output_complex.imag
    target_re = target_complex.real
    target_im = target_complex.imag
    original_mse_re = F.mse_loss(output_re, target_re)
    original_mse_im = F.mse_loss(output_im, target_im)
    original_complex_mse = original_mse_re + original_mse_im
    print(f"\nOriginal Complex MSE Loss: {original_complex_mse.item()}")

    loss_v1 = complex_relative_mse_loss_v1(output_complex, target_complex)
    loss_v2 = complex_relative_mse_loss_v2(output_complex, target_complex)
    loss_v3 = complex_relative_mse_loss_v3(output_complex, target_complex)

    print(f"Complex Relative MSE Loss V1: {loss_v1.item()}")
    print(f"Complex Relative MSE Loss V2: {loss_v2.item()}")
    print(f"Complex Relative MSE Loss V3: {loss_v3.item()}")

    # Test with all zero target for epsilon
    output_zero_target = torch.randn(2, dtype=torch.complex64)
    target_all_zeros = torch.zeros(2, dtype=torch.complex64)
    print("\n--- Testing with zero target ---")
    print(f"Output (zero target): {output_zero_target}")
    print(f"Target (zero target): {target_all_zeros}")

    loss_v1_zero = complex_relative_mse_loss_v1(output_zero_target, target_all_zeros)
    loss_v2_zero = complex_relative_mse_loss_v2(output_zero_target, target_all_zeros)
    loss_v3_zero = complex_relative_mse_loss_v3(output_zero_target, target_all_zeros)

    print(f"Complex Relative MSE Loss V1 (zero target): {loss_v1_zero.item()}")
    print(f"Complex Relative MSE Loss V2 (zero target): {loss_v2_zero.item()}")
    print(f"Complex Relative MSE Loss V3 (zero target): {loss_v3_zero.item()}")