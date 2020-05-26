# Iterative Adversarial Perturbations against Personalized Latent Recommenders.
This is the official github repository of the paper **Iterative Adversarial Perturbations against Personalized Latent Recommenders**.

RecSys2020 Submission ID: 1435

**Table of Contents:**
- [Requirements](#requirements)
- [Reproduce paper results](#reproduce-paper-results)

## Requirements

To begin with, please make sure your system has these installed:

* Python 3.6.8
* CUDA 10.1
* cuDNN 7.6.4

Then, install all required Python dependencies with the command:
```
pip install -r requirements.txt
```

## Reproduce paper results
Here we describe the steps to reproduce the results presented in the paper.

### 1. Train BPR-MF recommender model
First of all, train the BPR-MF recommender model by running:
```
python train.py \
  --dataset <dataset_name> \
  --rec bprmf \
  --epochs 2000 \
  --k 10
 ```
 
 ### 2. Train AMF recommender model
The, train AMF model by running:
```
python train.py \
  --dataset <dataset_name> \
  --rec apr \
  --epochs 2000 \
  --restore_epochs 100 \
  --k 10
 ```
 
