# Streamlit app to deploy our custom learnable layer

In this folder, you will find a streamlit app where we deployed a FFTRadnet network from Radial Valeao github 
repo we forked:

https://github.com/Krysalide/RADIalP7

Our repo contains some additional code for computing range doppler maps.
Those range doppler maps were necessary for supervised learning of the FFT's.

Work of interest resides in this folder:

https://github.com/Krysalide/RADIalP7/tree/main/SignalProcessing

The Streamlit app demonstrates the possibilty to use a neural network that directly takes as inputs 
raw radar data. We noticed a long inference time possibly due to the fact the network was deployed in a streamlit app.

We also deployed a Yolo model able to do some inferences on the images of the Radial dataset.
It was easy to deploy and very accurate and fast in it's predictions.
It is able to both locate and classify targets.
The only negative point for such model is that they don't work at night or in bad weather condition. 

Note: the code might not been cleaned, we had to move to the second part of the project on birdclef dataset.

 
