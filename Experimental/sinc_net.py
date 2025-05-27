import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
from torch.autograd import Variable
import math

## Function That I was Talking about, used in SincNet
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)
    

    

def act_fun(act_type):
    """Function that converts dictionary String values to nn functions.

    Args:
        act_type (String): A string describing the activation function desired.

    Returns:
        nn.ActivationFunction: Returns the desired activation functions.
    """
    if act_type=="relu":
        return nn.ReLU()
            
    if act_type=="tanh":
        return nn.Tanh()
            
    if act_type=="sigmoid":
        return nn.Sigmoid()
            
    if act_type=="leaky_relu":
        return nn.LeakyReLU(0.2)
            
    if act_type=="elu":
        return nn.ELU()
                        
    if act_type=="softmax":
        return nn.LogSoftmax(dim=1)
        
    if act_type=="linear":
        return nn.LeakyReLU(1) # initializzed like this, but not used in forward!
    
    ## Added SeLu and PReLu   
    if act_type=="selu": 
        return nn.SELU()

    if act_type=="prelu":
        return nn.PReLU(init=0.2) #Initial value of alpha is 0.2 like in LeakyReLu above


            
## Layer that normalizes input in SincNet and MLP
class LayerNorm(nn.Module):
    """
        This class is used by the standard SincNet for layernorm. It is inherited by Ravanelli's implementation.
        In SincNet2D we decided to switch to the version pytorch offered. See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    ## Normalizes the info
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    
## First layer of SincNet:
# They used symetry of sinc in this one in contrary to SincConv:
class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        ## If user sets the initial input channel of SincConv to a value different from 1:
        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        ## For Generalisation purposes only...
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        ## Initializes the filters' bound :
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        # initialize filterbanks such that they are equally spaced in Mel scale
        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)

        # Converts back to Hz, no longer equally spaced.
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # Intializes the number of points to the left of the time axis:
        n = (self.kernel_size - 1) / 2.0

        # self.n_ = 2*pi*n/fs where n is in [|-(self.kernel_size)/2, -1|]:  
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axis

 


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        ## Sens the frequency periods and the window to the same device as the input:
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        ##Initializing low and high frequencies: 
        low = self.min_low_hz  + torch.abs(self.low_hz_)
        ## Here we force the high frequency to be superior to the low frequency and lower than fs/2 
        # see doc here https://pytorch.org/docs/stable/generated/torch.clamp.html
        # This contradicts what Ravanelli et al. said in their paper: 
        """ From SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET
        "Note that no bounds have been imposed to force f2 (high) to
        be smaller than the Nyquist frequency, since we observed that
        this constraint is naturally fulfilled during training."        
        """
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        
        ## Bandwith that is fed :
        band=(high-low)[:,0]
        
        ## Multiplies the low and high frequencies by 2*pi*n/fs
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        ## Computes the expression of the bandpass filter in time frame:
        # Left of the time axis:
        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.7 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). 
        # Bandpass for t=0s
        band_pass_center = 2*band.view(-1,1)
        # Right of the time axis:
        band_pass_right= torch.flip(band_pass_left,dims=[1])#We can do that because of symetry!

        ## <!> Adds up the three parts of the bandpass previously computed in order to have the complete expression here: <!> ##
        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)

        
        band_pass = band_pass / (2*band[:,None])
        

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        ## Here we use pytorch conv1d function to compute the convolution between the self.filters and the input: 
        # See https://pytorch.org/docs/stable/nn.functional.html?highlight=f%20conv1d#torch.nn.functional.conv1d
        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1) 

## Method used By Ravanelli before realising that the sinc function was symmetric around the y axis:
"""        
# Takes a bandwith and a time and returns a sinc tensor, is used in sinc_conv:
# @band    is a float 
# @t_right is a tensor
# t_right=Variable(torch.linspace(1, (N-1)/2, steps=int((N-1)/2))/self.fs).cuda()
def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y
        
class sinc_conv(nn.Module):

    def __init__(self, N_filt,Filt_dim,fs):
        super(sinc_conv,self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10**(mel_points / 2595) - 1)) # Convert Mel to Hz
        b1=np.roll(f_cos,1)
        b2=np.roll(f_cos,-1)
        b1[0]=30
        b2[-1]=(fs/2)-100
                
        self.freq_scale=fs*1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

        
        self.N_filt=N_filt
        self.Filt_dim=Filt_dim
        self.fs=fs
        

    def forward(self, x):
        
        filters=Variable(torch.zeros((self.N_filt,self.Filt_dim))).cuda()
        N=self.Filt_dim
        t_right=Variable(torch.linspace(1, (N-1)/2, steps=int((N-1)/2))/self.fs).cuda()
        
        
        min_freq=50.0;
        min_band=50.0;
        
        filt_beg_freq=torch.abs(self.filt_b1)+min_freq/self.freq_scale
        filt_end_freq=filt_beg_freq+(torch.abs(self.filt_band)+min_band/self.freq_scale)
       
        n=torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window=0.54-0.46*torch.cos(2*math.pi*n/N);
        window=Variable(window.float().cuda())

        
        for i in range(self.N_filt):
                        
            low_pass1 = 2*filt_beg_freq[i].float()*sinc(filt_beg_freq[i].float()*self.freq_scale,t_right)
            low_pass2 = 2*filt_end_freq[i].float()*sinc(filt_end_freq[i].float()*self.freq_scale,t_right)
            band_pass=(low_pass2-low_pass1)

            band_pass=band_pass/torch.max(band_pass)

            filters[i,:]=band_pass.cuda()*window

        out=F.conv1d(x, filters.view(self.N_filt,1,self.Filt_dim))
    
        return out
"""

## The main Network That is used by this Notebook to solve our speaker Recognition problem
class SincNet(nn.Module):
    def __init__(self,options):
       super(SincNet,self).__init__()
    
       self.cnn_N_filt=options['cnn_N_filt']
       self.cnn_len_filt=options['cnn_len_filt']
       self.cnn_max_pool_len=options['cnn_max_pool_len']
       self.use_SincConv_fast=options['use_SincConv_fast']
       
       
       self.cnn_act=options['cnn_act']
       self.cnn_drop=options['cnn_drop']
       
       self.cnn_use_laynorm=options['cnn_use_laynorm']
       self.cnn_use_batchnorm=options['cnn_use_batchnorm']
       self.cnn_use_laynorm_inp=options['cnn_use_laynorm_inp']
       self.cnn_use_batchnorm_inp=options['cnn_use_batchnorm_inp']
       
       self.input_dim=int(options['input_dim'])
       
       self.fs=options['fs']
       
       self.N_cnn_lay = len(options['cnn_N_filt'])
       self.conv      = nn.ModuleList([])
       self.bn        = nn.ModuleList([])
       self.ln        = nn.ModuleList([])
       self.act       = nn.ModuleList([])
       self.drop      = nn.ModuleList([])
       
             
       if self.cnn_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
           
       if self.cnn_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
       current_input=self.input_dim 
       
       for i in range(self.N_cnn_lay):
         
         N_filt=int(self.cnn_N_filt[i])
         len_filt=int(self.cnn_len_filt[i])
         
         # dropout
         self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
         
         # activation
         self.act.append(act_fun(self.cnn_act[i]))
                    
         # layer norm initialization         
         self.ln.append(LayerNorm([N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])]))

         self.bn.append(nn.BatchNorm1d(N_filt,int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i]),momentum=0.05))
            

         ## ! Very important uses SincConv_Fast as first layer(i==0)
         if i==0:
          if self.use_SincConv_fast:
           self.conv.append(SincConv_fast(self.cnn_N_filt[i],self.cnn_len_filt[i],self.fs))
          else:
           self.conv.append(nn.Conv1d(1, self.cnn_N_filt[i], self.cnn_len_filt[i]))   
         else:
          self.conv.append(nn.Conv1d(self.cnn_N_filt[i-1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
         
         ## Formula marker 1:
         current_input=int((current_input-self.cnn_len_filt[i]+1)/self.cnn_max_pool_len[i])

       ## output dimension of the network is computed dynamically
       self.out_dim=current_input*N_filt#N_filt=int(self.cnn_N_filt[-1])



    def forward(self, x):
       batch=x.shape[0]
       seq_len=x.shape[1]
       
       if bool(self.cnn_use_laynorm_inp):
        x=self.ln0((x))
        
       if bool(self.cnn_use_batchnorm_inp):
        x=self.bn0((x))
        
       x=x.view(batch,1,seq_len)

       
       for i in range(self.N_cnn_lay):
           
         if self.cnn_use_laynorm[i]:
          if i==0:
           x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))  
          else:
           x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))   
          
         if self.cnn_use_batchnorm[i]:
          x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

         if self.cnn_use_batchnorm[i]==False and self.cnn_use_laynorm[i]==False:
          x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

       # Flattens the tensor:       
       x = x.view(batch,-1)

       return x
   

## Imports used by SincNet2D:
import time
import matplotlib.pyplot as plt
import time

# This SincNet does not use anymore LayerNorm defined above bu Layernorm from pytorch: https://pytorch.org/docs/master/generated/torch.nn.LayerNorm.html
class SincNet2D(nn.Module):
    """Modified version of SincNet. 
    Uses Sinc-based convolution defined in SincConv_fast for the first layer of the CNN,
    then uses conv2d and maxpool2d for the rest of the layers of the CNN.

    Args:
        options (dict):    Takes as argument for initialization a dictionary named options.
        print_spec (bool): Second argument is a boolean that indicates if user wishes to print the spectrograms. Defaults to False.
    """
    @staticmethod
    def tensor_to_mel(tensor_hz):
        return 2595 * (1 + tensor_hz / 700).log10()
    
    @staticmethod
    def tensorLogScale(tensor):
        # Constant that translates all values to 10^-3
        eps = 1e-3
    
        # log(eps + pow):
        return (eps + tensor).log10()
    
    @staticmethod
    def PrintSpectrograms(specs, specType):
    
        N_audios = specs.size(0)
        N_column = 4

        fig, ax = plt.subplots(int(np.ceil(N_audios/N_column)), N_column, figsize=(14, 7),
                              subplot_kw={'xticks': [], 'yticks': []})   

        for i,spec in enumerate(specs):

            ## Position in the axes grid:
            axi, axj = int(i/N_column), i%N_column
            
            ## imshow only works on CPU tensors:
            if spec.is_cuda:
                spec = spec.cpu()
            
            # Here, we control the values of the ordinate and abscissa with the variable extent:
            pos = ax[axi][axj].imshow(spec, origin='lower', cmap='jet', aspect='auto')

            # Plot colorbar
            fig.colorbar(pos, ax=ax[axi][axj])
        
        # Places the spectrograms next to each other:
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        
        # Writes the title:
        fig.suptitle("Image representation of a " + specType + " Spectrogram for each audio file:", fontsize=20, y=1.1)

        plt.show()   

    @staticmethod
    def EnergyWindowMean(audios, N_filter = 80, L=300, stride=150, padding = False, removingZeros = False, debug_mode = False):

        if debug_mode:
            t1 = time.time()
        
        # First axe of audio is the time, second is the frequency band.
        batch_size = audios.size(0)
        N_filter   = audios.size(1)
        N          = audios.size(2)

        ## Just for display purposes, removes
        if(debug_mode):print(N)
            
        ## Storing if the network is in cuda:
        net_is_cuda = audios.is_cuda

        ## Adds padding if it is requested:
        if(padding):

            if((N - L)%stride != 0):

                ## Computing the new size with padding:
                new_size = N + stride - (N - L)%stride

                ## Padding in torch:
                target = torch.zeros(batch_size, N_filter, new_size, dtype = torch.float)
                
                ## If cuda is used, we do too:
                if net_is_cuda:
                    target = target.cuda()
                    
                target[:, :, :N] = audios
                audios = target

            ## Computes the new length after padding 
            N = audios.size(2)

            if(debug_mode):print(N-L, N)

        if debug_mode:
            t2 = time.time() 
            print("Temps de calcul pour le padding Energy : {}min".format( (t2-t1)/60 ))

        ## <!>--------- Computes the enrgy Here ---------<!> ##

        # Initializes the energy array
        #np.array([sum([el*el for el in audios[i:i+L]])/L  for i in (0, (N - L), stride)])
        Energy = torch.zeros(batch_size, N_filter, int((N-L)/stride) + 1, dtype = torch.float)
        
        ## If input is on cuda, we send the tensor to cuda:
        if net_is_cuda:
            Energy = Energy.cuda()
        
        
        if debug_mode:
            t3 = time.time() 
            print("Temps de calcul pour créer le tenseur Energy : {}min".format( (t3-t2)/60 ))
        
        ## Computing the energy of the signal:
        audios = audios.pow(2)
        
        
        if debug_mode:
            t4 = time.time() 
            print("Temps de calcul pour multiplier les matrices : {}min".format( (t4-t3)/60 )) 

        ## Very important +1 is needed for i to be equal to (N - L)!
        for i in range(0, (N - L + 1), stride):
            Energy[:, :, int(i/stride)] = audios[:, :, i:i+L].sum(dim=2)/L

                
        if debug_mode:
            t5 = time.time() 
            print("Temps de calcul pour le calcul de l'Energie : {}min".format( (t5-t4)/60 ))

        ## <!>------------------- Done -------------------<!> ##

        if(removingZeros):
            ## Removing zeros from energy:
            for i in range (len(Energy)-1, -1, -1):
                if(Energy[i] > 0):
                    Energy = Energy[:i+1]
                    break

        if(debug_mode):
            print(len(Energy))

            string_pad = "with padding" if padding else "without padding"

            print("Expected length of the array " + string_pad + " : " + str(int((N-L)/stride) + 1))

            t6 = time.time()
            print("Temps de calcul pour le reste : {}min".format( (t6-t5)/60 ))
                    
        
        return Energy
    
    def __init__(self,options, print_spec = False):
        super(SincNet2D,self).__init__()

        ## Plot parameters:
        self.print_spec = print_spec

        ## Parameters for convolutions:
        self.cnn_N_filt         = options['cnn_N_filt']
        self.cnn_len_filt_W     = options['cnn_len_filt_W']
        self.cnn_len_filt_H     = options['cnn_len_filt_H']
        self.cnn_max_pool_len_W = options['cnn_max_pool_len_W']
        self.cnn_max_pool_len_H = options['cnn_max_pool_len_H']
        self.use_SincConv_fast  = options['use_SincConv_fast']

        # Parameters used in order to compute the enrgy:
        self.cnn_energy_L      = options['cnn_energy_L']
        self.cnn_energy_stride = options['cnn_energy_stride']

        # Parameters for activation function and drop:
        self.cnn_act  = options['cnn_act']
        self.cnn_drop = options['cnn_drop']

        # Parameters for normalization:
        self.cnn_use_laynorm       = options['cnn_use_laynorm']
        self.cnn_use_batchnorm     = options['cnn_use_batchnorm']
        self.cnn_use_laynorm_inp   = options['cnn_use_laynorm_inp']
        self.cnn_use_batchnorm_inp = options['cnn_use_batchnorm_inp']

        # The size of the input:
        self.input_dim=int(options['input_dim'])

        # The sample rate:
        self.fs=options['fs']

        # Number of filters for each layer:
        self.N_cnn_lay=len(options['cnn_N_filt'])

        # Initialization of module lists:
        self.conv  = nn.ModuleList([])
        self.bn    = nn.ModuleList([])
        self.ln    = nn.ModuleList([])
        self.act   = nn.ModuleList([])
        self.drop  = nn.ModuleList([])

        # Input normalization layer:
        if self.cnn_use_laynorm_inp:
           self.ln0=nn.LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)

        ## 2D init:
        #Width and height of spectrograms for each layer:
        spec_H   = int(self.cnn_N_filt[0])
        spec_W   = int((self.input_dim -self.cnn_len_filt_W[0]+1)/self.cnn_max_pool_len_W[0])

        for i in range(self.N_cnn_lay):
         
            N_filt=int(self.cnn_N_filt[i])
        
            #2D iter:
            if i!=0:
                spec_H   = int((spec_H - self.cnn_len_filt_H[i] + 1) / self.cnn_max_pool_len_H[i])
                spec_W   = int((spec_W - self.cnn_len_filt_W[i]+1) / self.cnn_max_pool_len_W[i])
                if self.print_spec:
                    print(spec_H, spec_W)
                 
            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))
            
            
            # N_filt is equal to self.cnn_N_filt[i]
            ## ! Very important uses SincConv_Fast as first layer(i==0)
            if i==0:
                if self.use_SincConv_fast:
                    self.conv.append(SincConv_fast(N_filt, self.cnn_len_filt_W[i], self.fs))
                else:
                    self.conv.append(nn.Conv1d(1, self.cnn_N_filt[i], self.cnn_len_filt[i]))
                    
                # After the computation of the energy, the size changes to:
                spec_W=int(np.ceil((spec_W-self.cnn_energy_L)/self.cnn_energy_stride + 1))
                if self.print_spec: 
                    print(spec_H, spec_W)
              
            elif i==1:
                # Here Input channel size is one because we transitionned from a 3D tensor to a 4D tensor with one channel: (batch_size, Channel=1, Height, Width)
                self.conv.append(nn.Conv2d(1, N_filt, kernel_size = (self.cnn_len_filt_H[i], self.cnn_len_filt_W[i])))
            else:
                self.conv.append(nn.Conv2d(self.cnn_N_filt[i-1], N_filt, kernel_size = (self.cnn_len_filt_H[i], self.cnn_len_filt_W[i]) ))

            # layer norm initialization
            if i==0:
                self.ln.append(nn.LayerNorm([1, spec_H, spec_W]))
            else:
                self.ln.append(nn.LayerNorm([N_filt, spec_H, spec_W]))

            # Batchnorm initialization
            if i==0:
                self.bn.append(nn.BatchNorm2d(1, momentum=0.05))
            else:
                self.bn.append(nn.BatchNorm2d(N_filt, momentum=0.05))


        ## output dimension of the network is computed dynamically
        self.out_dim=N_filt*spec_H*spec_W



    def forward(self, x):
        batch=x.shape[0]
        seq_len=x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            
            if self.print_spec:
                t1 = time.time()
            
            x=self.ln0((x))
                        
            if self.print_spec:
                t2 = time.time() 
                print("Temps de calcul pour la inp layernorm : {}min".format( (t2-t1)/60 ))
            

        if bool(self.cnn_use_batchnorm_inp):
            x=self.bn0((x))

        x=x.view(batch,1,seq_len)

       
        for i in range(self.N_cnn_lay):
            
            # Taking time of conv:    
            if self.print_spec:
                t1 = time.time()
            
            # Conv:
            x = self.conv[i](x)
            
            if self.print_spec:
                t2 = time.time() 
                print("Temps de calcul pour la conv : {}min".format( (t2-t1)/60 ))
            
            ## Changes beacause of the torch.abs, I don't know why:
            if self.cnn_use_laynorm[i] and i==0:
                x = torch.abs(x)
            
            # Max pooling:
            if i==0:# Here we check the value of i because for i=0 the input is still in 2D.
                x = F.max_pool1d(x, kernel_size = self.cnn_max_pool_len_W[i])
            else:
                x = F.max_pool2d(x, kernel_size = (self.cnn_max_pool_len_H[i], self.cnn_max_pool_len_W[i]))
            
            if self.print_spec:
                t3 = time.time() 
                print("Temps de calcul pour maxpool : {}min".format( (t3-t2)/60 ))
            
            # Compute Energy with average pooling:                     
            if i == 0:
                # Taking time of Energy computation:
                if self.print_spec:
                    t3 = time.time()
                
                ## Computing how much zeros we need to add for the padding:
                # Size of the sequence:
                N = x.size(2)
                # Remaining zeros to add:
                to_pad = self.cnn_energy_stride - (N - self.cnn_energy_L)%self.cnn_energy_stride
                # Here we divide it by two in order to fit with the requirements of avg_pooling1d,
                # see https://pytorch.org/docs/stable/nn.functional.html#avg-pool1d
                to_pad /= 2
                to_pad = int(to_pad)
            
                # Computing the Energy:
                x = x.pow(2)
                x = F.avg_pool1d(x, kernel_size=self.cnn_energy_L, stride=self.cnn_energy_stride, padding=to_pad)
                
                if self.print_spec:
                    t4 = time.time() 
                    print("Temps de calcul pour l'Energie : {}min".format( (t4-t3)/60 ))
                
                # Converting the tensor to Log scale:
                x = self.tensorLogScale(x)

                # Printing the resulting logmel spectrograms:
                if self.print_spec:
                    self.PrintSpectrograms(x, "Log")
                
                ## <!> This is the moment when we switch from 1D to 2D <!> ##:
                x = x.view(batch, 1, self.cnn_N_filt[i], -1)


            # Layer normalization:
            if self.cnn_use_laynorm[i]:
                x = self.ln[i](x)
            # Batch norm:
            elif self.cnn_use_batchnorm[i]:
                x = self.bn[i](x)
            
            if self.print_spec:
                t5 = time.time() 
                print("Temps de calcul pour batchnorm : {}min".format( (t5-t4)/60 ))
            
            # activation function:                               
            x = self.act[i](x)
            
            if self.print_spec:
                t6 = time.time() 
                print("Temps de calcul pour la fonction d'activation : {}min".format( (t6-t5)/60 ))
            
            # Dropout layer:                              
            x = self.drop[i](x)    
            
            if self.print_spec:
                t7 = time.time() 
                print("Temps de calcul pour la fonction Drop : {}min".format( (t7-t6)/60 ))

            if self.print_spec:
                print(x.shape)

       
        x = x.view(batch,-1)

        return x

    
## MLP Generic Class
class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()
        
        self.input_dim            = int(options['input_dim'])
        self.fc_lay               = options['fc_lay']
        self.fc_drop              = options['fc_drop']
        self.fc_use_batchnorm     = options['fc_use_batchnorm']
        self.fc_use_laynorm       = options['fc_use_laynorm']
        self.fc_use_laynorm_inp   = options['fc_use_laynorm_inp']
        self.fc_use_batchnorm_inp = options['fc_use_batchnorm_inp']
        
        ## Activation functions given by the .cfg file under [dnn]
        self.fc_act = options['fc_act']
        
        ## Initializing module list:
        self.wx   = nn.ModuleList([])
        self.bn   = nn.ModuleList([])
        self.ln   = nn.ModuleList([])
        self.act  = nn.ModuleList([])
        self.drop = nn.ModuleList([])
       

       
        # input layer normalization
        if self.fc_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
          
        # input batch normalization    
        if self.fc_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d([self.input_dim],momentum=0.05)
           
           
        self.N_fc_lay=len(self.fc_lay)
             
        current_input=self.input_dim
        
        # Initialization of hidden layers
        
        for i in range(self.N_fc_lay):
            
         # dropout
         self.drop.append(nn.Dropout(p=self.fc_drop[i]))
         
         ## First MLP is leaky_relu,leaky_relu,leaky_relu
         ## Second is softmax
         # activation
         self.act.append(act_fun(self.fc_act[i]))
         
         
         add_bias=True
         
         # layer norm initialization
         self.ln.append(LayerNorm(self.fc_lay[i]))
         self.bn.append(nn.BatchNorm1d(self.fc_lay[i],momentum=0.05))
         
         if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
             add_bias=False
         
              
         # Linear operations
         self.wx.append(nn.Linear(current_input, self.fc_lay[i],bias=add_bias))
         
         # weight initialization
         self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.fc_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.fc_lay[i])),np.sqrt(0.01/(current_input+self.fc_lay[i]))))
         self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))
         
         current_input=self.fc_lay[i]
         
         
    def forward(self, x):
        
      # Applying Layer/Batch Norm
      if bool(self.fc_use_laynorm_inp):
        x=self.ln0((x))
        
      if bool(self.fc_use_batchnorm_inp):
        x=self.bn0((x))
        
      for i in range(self.N_fc_lay):

        if self.fc_act[i]!='linear':
            
          if self.fc_use_laynorm[i]:
           x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
          
          if self.fc_use_batchnorm[i]:
           x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
          
          if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
           x = self.drop[i](self.act[i](self.wx[i](x)))
           
        else:
          if self.fc_use_laynorm[i]:
           x = self.drop[i](self.ln[i](self.wx[i](x)))
          
          if self.fc_use_batchnorm[i]:
           x = self.drop[i](self.bn[i](self.wx[i](x)))
          
          if self.fc_use_batchnorm[i]==False and self.fc_use_laynorm[i]==False:
           x = self.drop[i](self.wx[i](x)) 
          
      return x


## Class that contains all the networks:
class MainNet(nn.Module):
    
    def __init__(self, CNN_net, DNN1_net, DNN2_net):
        super(MainNet, self).__init__()
        self.CNN_net  = CNN_net
        self.DNN1_net = DNN1_net
        self.DNN2_net = DNN2_net

    def forward(self, x):
        x = self.DNN2_net(self.DNN1_net(self.CNN_net(x)))
        return x
