Dataset Description：
This repository contains the original and preprocessed data used in the paper, including:
Commercial Microwave Links (CML): Received signal levels (RSL) in dBm.
Rain Gauge (RG) Measurements: Ground-truth rainfall intensity (mm).

Data Format:
File format: .xlsx (Excel)
Timestamp format: yyyy-mm-dd HH:MM:SS

Data versions:
Original data: Irregular time series (raw measurements).
Preprocessed data: Regular time series resampled to 1-minute temporal resolution.

Code Instructions:
Dependencies
Python 3.8
PyTorch 1.11.0
CUDA 11.3 (For GPU acceleration)
Trained on NVIDIA GeForce RTX 3090
⚠️ Note: Results may vary slightly with different versions of PyTorch/CUDA.

Reproduction Steps:
1) Data Preprocessing.
Run the preprocessing script to generate regular time series.

3) Continuous Wavelet Transform (CWT).
Execute cwt.py to convert CML attenuation time series into scalograms.

4) Class Balancing.
Run random_sampling.py to balance the number of wet (rainy) and dry samples in the training set.

5) Model Training & Evaluation.
Execute cnn_attention.py to train the CNN+CA (Convolutional Neural Network with Channel Attention) model and obtain wet/dry classification results.

Important Notes:
Path Configuration:
The storage paths for generated files may differ between scripts due to separate Python interpreter runs. Modify paths in the scripts if needed.

GPU Requirement:
The model is optimized for GPU training. CPU execution may significantly slow down processing.
