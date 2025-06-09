import streamlit as st
import os
from Experimental.data_fft_reader import RadarFFTDataset
from data_reader import split_dataloader
st.title('DATASET PAGE')

data_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/TEST'
assert os.path.exists(data_folder), 'data not found'
element_number=60
assert element_number<61, f"number of element is limited to 60"
indices = list(range(element_number)) # specify number of elements

dataset = RadarFFTDataset(data_folder, indices)
print(f"Dataset length: {len(dataset)} (took only {element_number} samples)")

# train_loader, val_loader, test_loader = split_dataloader(dataset,batch_size=1,train_ratio=0.95,val_ratio=0.04,test_ratio=0.01)
# print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

test_indice=5

raw_adc_test=dataset.get_adc(idx=test_indice)
print(type(raw_adc_test))
print(raw_adc_test.shape)
range_doppler_test=dataset.get_range_doppler(idx=test_indice)
print(type(range_doppler_test))
print(range_doppler_test.shape)

antenna=5
print(10*'#')
antenna_view=raw_adc_test[:,:,antenna]
print(antenna_view.shape)
doppler_view=range_doppler_test[:,:,antenna]
print(doppler_view.shape)
print(10*'"')





