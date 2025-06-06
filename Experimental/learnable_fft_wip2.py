import torch
import torch.nn as nn
import numpy as np

'''
Last up to date version
'''

# TODO add windowing!! inside FFTLinearLayer

class FirstFFTLinearLayer(nn.Module):
    def __init__(self, input_size):
        super(FirstFFTLinearLayer, self).__init__()
        self.input_size = input_size

        # Create the DFT matrix as it is in custom_signal_processing
        # weights fit with fft matrix coefficients
        N = input_size
        j, k = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        dft_matrix_np = np.exp(-2j * np.pi * j * k / N)
        self.dft_matrix = nn.Parameter(torch.tensor(dft_matrix_np, dtype=torch.complex64), requires_grad=False)

        
        self.linear = nn.Linear(2 * input_size, 2 * input_size, bias=False)

        
        weight_matrix = torch.cat((
            torch.cat((self.dft_matrix.real, -self.dft_matrix.imag), dim=1),
            torch.cat((self.dft_matrix.imag, self.dft_matrix.real), dim=1)
        ), dim=0)
        
        self.linear.weight = nn.Parameter(weight_matrix.float(), requires_grad=True)
        self.linear.bias = None

    def forward(self, complex_adc):
        
    
        # Permute the input to bring the FFT dimension to the last
        # We want the FFT to be applied along the second dimension (index 1)
        input_reshaped = complex_adc.permute(0, 2, 3, 1) 

        # Flatten the non-FFT dimensions
        flattened_shape = (-1, self.input_size)
        input_flattened = input_reshaped.reshape(flattened_shape) 

        real_input = input_flattened.real
        imag_input = input_flattened.imag
        combined_input = torch.cat((real_input, imag_input), dim=-1) 

        # Perform the linear transformation
        combined_output = self.linear(combined_input)

        # Separate real and imaginary parts of the output
        # they were previously concatenated
        real_output = combined_output[..., :self.input_size] 
        imag_output = combined_output[..., self.input_size:]

        # Combine back into a complex tensor
        output_complex_flattened = torch.complex(real_output, imag_output) 

        # Reshape back to the intermediate shape
        output_reshaped = output_complex_flattened.reshape(input_reshaped.shape) 

        # Permute back to the original shape with the FFT dimension in the correct place
        return output_reshaped.permute(0, 3, 1, 2) # (batch, 512, 256, 16)
    

    # experimental, to see weights 
    def get_range_fft_weights(self):
        return self.linear.weight
    
# same as FirstFFTLayer except the fft is done one the third dimensions
# size of fft differ (256 not 512)
class SecondFFTLinearLayer(nn.Module):
    def __init__(self, input_size):
        super(SecondFFTLinearLayer, self).__init__()
        self.input_size = input_size

        # Create the DFT matrix as it is in custom_signam_processing
        N = input_size
        j, k = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        dft_matrix_np = np.exp(-2j * np.pi * j * k / N)
        self.dft_matrix = nn.Parameter(torch.tensor(dft_matrix_np, dtype=torch.complex64), requires_grad=False)

        #
        self.linear = nn.Linear(2 * input_size, 2 * input_size, bias=False)

        # Initialize the weight matrix of the linear layer with the DFT matrix
        weight_matrix = torch.cat((
            torch.cat((self.dft_matrix.real, -self.dft_matrix.imag), dim=1),
            torch.cat((self.dft_matrix.imag, self.dft_matrix.real), dim=1)
        ), dim=0)

        self.linear.weight = nn.Parameter(weight_matrix.float(), requires_grad=True)
        self.linear.bias = None

    def forward(self, complex_adc):

        # We want the FFT to be applied along the third dimension (index 2)
        
        input_reshaped = complex_adc.permute(0, 1, 3, 2) # (batch, 512, 16, 256) (batch, C, W, H)

        # Flatten the non-FFT dimensions
        
        flattened_shape = (-1, self.input_size)
        input_flattened = input_reshaped.reshape(flattened_shape) 

        real_input = input_flattened.real
        imag_input = input_flattened.imag
        combined_input = torch.cat((real_input, imag_input), dim=-1) 

        # Perform the linear transformation
        combined_output = self.linear(combined_input) 

        # Separate real and imaginary parts of the output
        real_output = combined_output[..., :self.input_size] 
        imag_output = combined_output[..., self.input_size:] 

        # Combine back into a complex tensor
        output_complex_flattened = torch.complex(real_output, imag_output) 

        # Reshape back to the intermediate shape
        output_reshaped = output_complex_flattened.reshape(input_reshaped.shape) 

        # Permute back to the original shape with the FFT dimension in the correct place
        
        return output_reshaped.permute(0, 1, 3, 2) 
    
    def get_doppler_fft_weights(self):
        return self.linear.weight
    
class Hamming_window_range(nn.Module):
    '''
    applies windowing as it is found in radial repo
    if not applied results can differ in an important way
    for now we have to set this layer to non trainable? 
    '''

    def __init__(self):
        super(Hamming_window_range, self).__init__()
        self.hanning_window_range=torch.tensor(np.load('/home/christophe/ComplexNet/Experimental/hanning_window_range.npy'))
    def forward(self,complex_adc):
        windowed_signal=torch.multiply(complex_adc,self.hanning_window_range.to('cuda'))
        return windowed_signal
    
    def get_window_range_coefficients(self):
        print(self.hanning_window_range.shape)
        return self.hanning_window_range



class Hamming_window_doppler(nn.Module):
    def __init__(self):
        super(Hamming_window_doppler, self).__init__()
        self.hanning_window_doppler=torch.tensor(np.load('/home/christophe/ComplexNet/Experimental/hanning_window_dopller.npy'))
    def forward(self,complex_adc):
        windowed_signal=torch.multiply(complex_adc,self.hanning_window_doppler.to('cuda'))
        return windowed_signal
    
    def get_window_doppler_coefficients(self):
        print(self.hanning_window_doppler.shape)
        return self.hanning_window_doppler
    
# ChatGPT Code does not work ???
# class Hamming_window_rangeV2(nn.Module):
#     def __init__(self):
#         super(Hamming_window_range, self).__init__()
#         window = np.load('/home/christophe/ComplexNet/Experimental/hanning_window_range.npy')
#         self.register_buffer('hanning_window_range', torch.tensor(window, dtype=torch.complex64))

#     def forward(self, complex_adc):
#         return complex_adc * self.hanning_window_range
    

class SignalProcessLayer(nn.Module):
    name='signal_process_neural_network'
    def __init__(self):
        super().__init__()
        self.hamming1 = Hamming_window_range()
        self.first_fft_layer=FirstFFTLinearLayer(input_size=512)
        self.hamming2=Hamming_window_doppler()
        self.second_fft_layer=SecondFFTLinearLayer(input_size=256)

    
    def forward(self, x):
        x = self.hamming1(x)
        x=x.clone().detach().to(dtype=torch.complex64)
        x = self.first_fft_layer(x)
        x=self.hamming2(x)
        x=x.clone().detach().to(dtype=torch.complex64)
        x=self.second_fft_layer(x)
        return x
    
    def get_range_weights(self):
        return self.first_fft_layer.get_range_fft_weights()
    
    def get_doppler_weights(self):
        return self.second_fft_layer.get_doppler_fft_weights()
    def get_window_range_coeff(self):
        return self.hamming1.get_window_range_coefficients()
    def get_window_doppler_coeff(self):
        return self.hamming2.get_window_doppler_coefficients()

# does not contain windowing so useless in that form
def build_fft_by_dot_product_numpy(complex_adc):
    signal_windowed = complex_adc # Assuming range_fft_coef is all ones
    N = signal_windowed.shape[0] # FFT along the first dimension
    j, k = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    dft_matrix = np.exp(-2j * np.pi * j * k / N)
    range_fftv2 = np.tensordot(dft_matrix, signal_windowed, axes=([1], [0]))
    return range_fftv2


if __name__ == '__main__':

    hanning_window_range=Hamming_window_range()
   
    hanning_window_range.eval()
    
    fft_layer = FirstFFTLinearLayer(input_size=512) # 512 is the range fft number
    fft_layer.eval()

    hanning_window_doppler=Hamming_window_doppler()

    hanning_window_doppler.eval()

   
    fft_layer_2=SecondFFTLinearLayer(input_size=256) # 256 is the doppler fft number
    fft_layer_2.eval()
    print(30*'#')
    print('layers succesfully created')

    # raw data
    adc_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST/ADC/'
    # first fft computed with radial tools (ground truth)
    fft_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST/FFT/'
    # range doppler computed with radial tools (ground truth)
    fft_fold2=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST/FFT2/'

    sample_adc=np.load(adc_folder+'raw_adc_2.npy')
    # add batch dimension
    batch_sample_adc=torch.tensor(np.expand_dims(sample_adc, axis=0),dtype=torch.complex64)
    assert np.allclose(batch_sample_adc[0],sample_adc),"raw adc data corrupted while expanding"

    windowed_signal = hanning_window_range(batch_sample_adc).clone().detach().to(dtype=torch.complex64)

    # sort of ground truth, computed in dataset_maker file
    sample_fft=np.load(fft_folder+'first_fft_map_2.npy')
    batch_sample_fft=np.expand_dims(sample_fft,axis=0)
    assert np.allclose(batch_sample_fft[0],sample_fft),"fft data corrupted while expanding fft tensor"

    range_doppler=np.load(fft_fold2+'second_fft_2.npy')
    

    radar_data_output= fft_layer(windowed_signal)

    #windowed_signal = hanning_window_range(batch_sample_adc).clone().detach().to(dtype=torch.complex64)
    windowed_signal_2=hanning_window_doppler(radar_data_output).clone().detach().to(dtype=torch.complex64)

    final_range_dopler_frame=fft_layer_2(windowed_signal_2)

    bias_final=final_range_dopler_frame[0].detach().numpy()-range_doppler

    mean_difference = np.mean(np.abs(bias_final))
    print(f"Mean Absolute Difference final: {mean_difference}")
    max_magnitude=np.max(np.abs(bias_final))

    print('max magnitude difference final: ',max_magnitude)


    radar_data_output_numpy=radar_data_output[0].detach().numpy()


    print(30*'#')
    # bias shall always be small:
    bias2=sample_fft-radar_data_output_numpy
    assert np.allclose(sample_fft,radar_data_output_numpy,atol=0.001,rtol=0.001)," missmatch between ground truth and cnn output"

    mean_difference = np.mean(np.abs(bias2))
    print(f"Mean Absolute Difference between fourier layer and classical fft: {mean_difference}")

    max_magnitude=np.max(np.abs(bias2))

    print('max magnitude difference: ',max_magnitude)
    print(30*'#')

    print(30*'~')
    print('test of final model')
    signal_process_layer=SignalProcessLayer()
    signal_process_layer.eval()
    signal_processed=signal_process_layer(batch_sample_adc)
    print('Final shape: ',signal_processed.shape)
    bias_final=signal_processed[0].detach().numpy()-range_doppler

    mean_difference = np.mean(np.abs(bias_final))
    print(f"Mean Absolute Difference final with 4 layers model: {mean_difference}")
    max_magnitude=np.max(np.abs(bias_final))

    print('max magnitude difference final with 4 layers model: ',max_magnitude)
    


