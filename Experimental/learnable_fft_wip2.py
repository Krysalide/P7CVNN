import torch
import torch.nn as nn
import numpy as np

'''
Last up to date version
'''

# TODO add windowing!! inside FFTLinearLayer

class FFTLinearLayerV2(nn.Module):
    def __init__(self, input_size):
        super(FFTLinearLayerV2, self).__init__()
        self.input_size = input_size

        

        # Create the DFT matrix as it is in custom_signam_processing
        N = input_size
        j, k = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        dft_matrix_np = np.exp(-2j * np.pi * j * k / N)
        self.dft_matrix = nn.Parameter(torch.tensor(dft_matrix_np, dtype=torch.complex64), requires_grad=False)

        # Define a single linear layer with double the input and output size
        # to handle real and imaginary parts together
        self.linear = nn.Linear(2 * input_size, 2 * input_size, bias=False)

        # Initialize the weight matrix of the linear layer with the DFT matrix
        weight_matrix = torch.cat((
            torch.cat((self.dft_matrix.real, -self.dft_matrix.imag), dim=1),
            torch.cat((self.dft_matrix.imag, self.dft_matrix.real), dim=1)
        ), dim=0)
        #self.linear.weight = nn.Parameter(weight_matrix.float(), requires_grad=False)
        self.linear.weight = nn.Parameter(weight_matrix.float(), requires_grad=True)
        self.linear.bias = None

    def forward(self, complex_adc):
        # Get the original shape
        original_shape = complex_adc.shape
        batch_size = original_shape[0]
        

        # Permute the input to bring the FFT dimension to the last
        # We want the FFT to be applied along the second dimension (index 1)
        input_reshaped = complex_adc.permute(0, 2, 3, 1) # (batch, 256, 16, 512)

        # Flatten the non-FFT dimensions
        flattened_shape = (-1, self.input_size)
        input_flattened = input_reshaped.reshape(flattened_shape) # (batch * 256 * 16, 512)

        real_input = input_flattened.real
        imag_input = input_flattened.imag
        combined_input = torch.cat((real_input, imag_input), dim=-1) # (batch * 256 * 16, 1024)

        # Perform the linear transformation
        combined_output = self.linear(combined_input) # (batch * 256 * 16, 1024)

        # Separate real and imaginary parts of the output
        real_output = combined_output[..., :self.input_size] # (batch * 256 * 16, 512)
        imag_output = combined_output[..., self.input_size:] # (batch * 256 * 16, 512)

        # Combine back into a complex tensor
        output_complex_flattened = torch.complex(real_output, imag_output) # (batch * 256 * 16, 512)

        # Reshape back to the intermediate shape
        output_reshaped = output_complex_flattened.reshape(input_reshaped.shape) # (batch, 256, 16, 512)

        # Permute back to the original shape with the FFT dimension in the correct place
        return output_reshaped.permute(0, 3, 1, 2) # (batch, 512, 256, 16)
    
class Hamming_window_range(nn.Module):
    def __init__(self):
        super(Hamming_window_range, self).__init__()
        self.hanning_window_range=torch.tensor(np.load('/home/christophe/ComplexNet/Experimental/hanning_window_range.npy'))
    def forward(self,complex_adc):
        windowed_signal=torch.multiply(complex_adc,self.hanning_window_range)
        return windowed_signal


class Hamming_window_doppler(nn.Module):
    def __init__(self):
        super(Hamming_window_doppler, self).__init__()
        self.hanning_window_doppler=torch.tensor(np.load('/home/christophe/ComplexNet/Experimental/hanning_window_dopller.npy'))
    def forward(self,complex_adc):
        windowed_signal=torch.multiply(complex_adc,self.hanning_window_doppler)
        return windowed_signal



if __name__ == '__main__':

    
    
    # batch_size = 8
    # input_shape = (batch_size, 512, 256, 16)
    # complex_adc_np = np.random.rand(*input_shape) + 1j * np.random.rand(*input_shape)
    # complex_adc_tensor = torch.tensor(complex_adc_np, dtype=torch.complex64)
    hanning_window_range=Hamming_window_range()
    hanning_window_range.eval()
    
    fft_layer = FFTLinearLayerV2(input_size=512) # input_size=512
    fft_layer.eval()

    han_test=Hamming_window_doppler()

    # output_fft_tensor = fft_layer(complex_adc_tensor)
    # print("Input shape:", complex_adc_tensor.shape)
    # print("Output shape:", output_fft_tensor.shape)

    # pred=output_fft_tensor[0]
    # adc_sample=complex_adc_tensor[0]
    def build_fft_by_dot_product_numpy(complex_adc):
        signal_windowed = complex_adc # Assuming range_fft_coef is all ones
        N = signal_windowed.shape[0] # FFT along the first dimension
        j, k = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        dft_matrix = np.exp(-2j * np.pi * j * k / N)
        range_fftv2 = np.tensordot(dft_matrix, signal_windowed, axes=([1], [0]))
        return range_fftv2

    # numpy_fft_output = build_fft_by_dot_product_numpy(adc_sample)
    # pytorch_fft_output_detached = pred.detach().numpy()

    # print("Output shapes are the same:", numpy_fft_output.shape == pytorch_fft_output_detached.shape)
    # difference = numpy_fft_output - pytorch_fft_output_detached



    # # Mean Difference (element-wise)
    # mean_difference = np.mean(np.abs(difference))
    # print(f"Mean Absolute Difference: {mean_difference}")

    # max_magnitude=np.max(np.abs(difference))

    # print('max magnitude difference: ',max_magnitude)



    # # Magnitude Difference
    # magnitude_numpy = np.abs(numpy_fft_output)
    # magnitude_pytorch = np.abs(pytorch_fft_output_detached)
    # magnitude_difference = magnitude_numpy - magnitude_pytorch
    # mean_magnitude_difference = np.mean(np.abs(magnitude_difference))
    # print(f"Mean Absolute Magnitude Difference: {mean_magnitude_difference}")

    # # Phase Difference
    # phase_numpy = np.arctan2(numpy_fft_output.imag, numpy_fft_output.real)
    # phase_pytorch = np.arctan2(pytorch_fft_output_detached.imag, pytorch_fft_output_detached.real)
    # phase_difference = phase_numpy - phase_pytorch

    # # Wrap phase difference to [-pi, pi]
    # phase_difference = np.arctan2(np.sin(phase_difference), np.cos(phase_difference))

    # max_phase_difference=np.max(phase_difference)
    # print('max phase edifference: ',max_phase_difference)
    # mean_phase_difference = np.mean(np.abs(phase_difference))
    # print(f"Mean Absolute Phase Difference: {mean_phase_difference} radians")
    


    #adc_folder='/home/christophe/RADIalP7/SMALL_DATASET/RECORD@2020-11-22_12.08.31/ADC/'
    adc_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST/ADC/'
    #fft_folder='/home/christophe/RADIalP7/SMALL_DATASET/RECORD@2020-11-22_12.08.31/FFT/'
    fft_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST/FFT/'

    fft_fold2=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST/FFT2/'
    sample_adc=np.load(adc_folder+'raw_adc_2.npy')
    batch_sample_adc=torch.tensor(np.expand_dims(sample_adc, axis=0),dtype=torch.complex64)
    
    windowed_signal=torch.tensor(hanning_window_range(batch_sample_adc),dtype=torch.complex64)

    assert np.allclose(batch_sample_adc[0],sample_adc)
    sample_fft=np.load(fft_folder+'first_fft_map_2.npy')
    batch_sample_fft=np.expand_dims(sample_fft,axis=0)
    assert np.allclose(batch_sample_fft[0],sample_fft)

    range_doppler=np.load(fft_fold2+'second_fft_2.npy')

    radar_data_output= fft_layer(windowed_signal)
    radar_data_output=radar_data_output[0].detach().numpy()

    fft_by_dot_product=build_fft_by_dot_product_numpy(sample_adc)

    bias1=fft_by_dot_product-sample_fft

    mean_difference = np.mean(np.abs(bias1))
    print(f"Mean Absolute Difference 1: {mean_difference}")

    bias2=sample_fft-radar_data_output
    mean_difference = np.mean(np.abs(bias2))
    print(f"Mean Absolute Difference 2: {mean_difference}")

    max_magnitude=np.max(np.abs(bias2))

    print('max magnitude difference2: ',max_magnitude)

    bias3=fft_by_dot_product-radar_data_output
    mean_difference = np.mean(np.abs(bias3))
    print(f"Mean Absolute Difference 3: {mean_difference}")
    max_magnitude=np.max(np.abs(bias3))

    print('max magnitude difference 3: ',max_magnitude)

    print('')
