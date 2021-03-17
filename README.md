# MSAP: Multi-Step Adversarial Perturbations on Recommender Systems Embeddings

GitHub repository of the FLAIRS2021 paper: MSAP: Multi-Step Adversarial Perturbations on Recommender Systems Embeddings, published by Vito Walter Anelli, Alejandro Bellogín, Yashar Deldjoo, Tommaso Di Noia, and Felice Antonio Merra.

Paper available at [Sisinflab](http://sisinflab.poliba.it/publications/2021/ABDDM21/) publications web page.

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
  --k 10 \
  --adv_type fgsm\
  --adv_eps 0.5\
  --adv_reg 1
 ```
 ```adv_type```, ```adv_eps```, ```adv_reg``` are parameters set to specify the type of fgsm-like attack used to apply the adversarial regularization.
 
### 3. Run Attacks
Based upon the produced recommender model we can run the attacks:
```
python run_attack.py \
  --dataset <dataset_name> \
  --rec <recommendr_name> \
  --attack_type <attack_type> \
  --attack_eps <attack_eps> \
  --attack_step_size <attack_step_size> \
  --attack_iteration <attack_iteration> \
  --best 1
```
where ```attack_type``` can be ```[fgsm, bim, pgd]```, ```attack_eps``` is the budget perturbation (epsilon), ```attack_step_size``` is the step size (e.g., 4) used in the iterative attacks, ```attack_iteration``` is the number of iterations.

### 4. Recommendation Evaluation

The attack results are store as recommendation lists under the directory ```./rec_results/<dataset_name>/<model_name>/file_name.tsv```. Each result file can be evaluated with different evaluation frameworks.

## Authors Team
* Vito Walter Anelli (vitowalter.anelli@poliba.it)
* Alejandro Bellogín (alejandro.bellogin@uam.es)
* Yashar Deldjoo (yashar.deldjoo@poliba.it)
* Tommaso Di Noia (tommaso.dinoia@poliba.it)
* Felice Antonio Merra<sup id="a1">[*](#f1)</sup> (felice.merra@poliba.it) 

<b id="f1"><sup>*</sup></b> Corresponding author
