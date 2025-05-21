import torch

print(torch.__version__)
import neuralop

from neuralop.models import FNO2d

fno=FNO2d(complex_data=True,in_channels=16,out_channels=16,n_modes_height=516,n_modes_width=256,hidden_channels=1)


