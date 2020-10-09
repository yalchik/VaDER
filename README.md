This repository contains code for a new method for clustering multivariate short time series with potentially many missing values (published [here](https://academic.oup.com/gigascience/article/8/11/giz134/5626377)), a setting commonly encountered in the analysis of longitudinal clinical data, but generally still poorly addressed in the literature. The method is based on a variational autoencoder with a Gaussian mixture prior (with a latent loss as described [here](https://arxiv.org/abs/1611.05148)), extended with LSTMs for modeling multivariate time series, as well as implicit imputation and loss re-weighting for directly dealing with (potentially many) missing values.

The code was written using

(1) Python 3.6 and Tensorflow 1.10.1 (directory tensorflow1), and

(2) Python 3.8 and Tensorflow 2.3.1 (directory tensorflow2).

<hr>

### How to start
1. Clone the repository
2. Run ```pip install --editable .``` in the project directory
3. Import the VaDER model in your python code: ```from vader import VADER```
4. See a usage example in ```VaDER/tensorflow2/test/test_vader.py```
