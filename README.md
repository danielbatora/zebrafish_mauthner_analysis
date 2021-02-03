# zebrafish_mauthner_analysis
 
Functions used for the analysis of iGluSnFR and GCaMP linescan recordings at the Mauthner-cell of zebrafish larvae. 
The data was recorded with a Femtonics 2D two-photon microscope. 

The analysis included the following steps: 
 1. Extracting individuals recordings from MatLab to .csv file to access it with Python
 2. Loading, and manual selection of regions of interest. 
 3. Motion correction and fluorescent trace alignment using cross correlation algorithm
 4. Spike detection and deconvolution using two exponential fits
 
