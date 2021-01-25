This repository contains code for a method for clustering multivariate time series with potentially many missing values (published [here](https://academic.oup.com/gigascience/article/8/11/giz134/5626377)), a setting commonly encountered in the analysis of longitudinal clinical data, but generally still poorly addressed in the literature. The method is based on a variational autoencoder with a Gaussian mixture prior (with a latent loss as described [here](https://arxiv.org/abs/1611.05148)), extended with LSTMs for modeling multivariate time series, as well as implicit imputation and loss re-weighting for directly dealing with (potentially many) missing values.

The use of the method is not restricted to clinical data. It can generally be used for multivariate time series data. 

In addition to variational autoencoders with gaussian mixture priors, the code allows to train ordinary variational autoencoders (multivariate gaussian prior) and ordinary autoencoders (without prior), for all available time series models (LSTM, GRUs and Transformers).

The code was written using

(1) Python 3.6 and Tensorflow 1.10.1 (directory tensorflow1), and

(2) Python 3.8 and Tensorflow 2.4.0 (directory tensorflow2).

Note that only the Tensorflow 2.4.0 version gives the option for training transformer networks in addition to LSTMs/GRUs.

<hr>

## Installation
1. Clone the repository.
2. Run ```pip install .``` in the project directory.

## Usage
The model can be used in 2 ways: via CLI (```<project>/tools/run_vader.py```) and programmatically. In both ways, it requires certain hyperparameters as input data, so you can preliminarily run hyperparameters optimization via CLI (```<project>/tools/hyperparameters_optimization.py```)

For the programmatic usage, just import the model in your code (e.g. ```from vader import VADER```) and create a ```VADER``` object. You can find an example in the following unit-test: ```<project>/tensorflow2/test/test_vader.py``` or in the CLI script: ```<project>/tools/run_vader.py```.

### Hyperparameters optimization (preparation)
Before you can run the hyperparameters optimization, 2 requirements must be satisfied:

#### 1. Data function
There must be a function that transforms input data into a tensor;

Out of the box, ```hyperparameters_optimization.py``` script supports the following data types:
1. Normalized ADNI data (```input_data_type=ADNI```)
2. Non-normalized NACC data (```input_data_type=NACC```)

For any other data types, you need to implement the function ```read_custom_data``` in the ```<project>/tools/hyperparameters_optimization.py``` script and use the ```custom``` data type as the corresponding CLI argument.

#### 2. Parameters dictionary.
You need to provide a parameter dictionary, which defines ranges of possible values for each hyerparameter.

Out-of-box, there is ```PlainParamGridFactory``` class in the ```hyperparameters_optimization.py``` script, where you can just edit corresponding hyperparameters values. If you need more sophisticated behaviour, you can create your own sub-class of ```ParamGridFactory``` class and pass it to the ```VADERHyperparametersOptimizer``` initializer. Lastly, you can create and pass an object of ```ParamGridFactory``` which will provide the parameter grid used to reproduce the paper results.

#### Hyperparameters optimization CLI
**Usage**
```shell
python hyperparameters_optimization.py [-h] --input_data_file INPUT_DATA_FILE --input_data_type {ADNI,NACC,PPMI,custom} [--input_weights_file INPUT_WEIGHTS_FILE]
                                       [--input_seed INPUT_SEED] [--n_repeats N_REPEATS] [--n_proc N_PROC] [--n_sample N_SAMPLE] [--n_consensus N_CONSENSUS] [--n_epoch N_EPOCH]
                                       [--n_splits N_SPLITS] [--n_perm N_PERM] --output_folder OUTPUT_FOLDER

optional arguments:
  -h, --help            show this help message and exit
  --input_data_file INPUT_DATA_FILE
                        .csv file with input data
  --input_data_type {ADNI,NACC,PPMI,custom}
  --input_weights_file INPUT_WEIGHTS_FILE
                        .csv file with flags for missing values
  --input_seed INPUT_SEED
                        used both as KFold random_state and VaDER seed
  --n_repeats N_REPEATS
                        number of repeats, default 10
  --n_proc N_PROC       number of processor units that can be used, default 6
  --n_sample N_SAMPLE   number of hyperparameters set per CV, default - full grid
  --n_consensus N_CONSENSUS
                        number of repeats for consensus clustering, default 1
  --n_epoch N_EPOCH     number of epochs for VaDER training, default 10
  --n_splits N_SPLITS   number of splits in KFold per optimization job, default 2
  --n_perm N_PERM       number of permutations for prediction strength, default 100
  --output_folder OUTPUT_FOLDER
                        a directory where report will be written
```
The script result will be represented as a PDF report written in a given```output_folder```.

**Example (local smoke test):**
```shell
python hyperparameters_optimization.py --input_data_file=../vader_data/ADNI/Xnorm.csv
                                       --input_data_type=ADNI
                                       --n_repeats=3
                                       --n_sample=3
                                       --n_consensus=1
                                       --n_epoch=10
                                       --n_splits=2
                                       --n_perm=10
                                       --output_folder=../vader_hp_opt_results_smoke
```

**Example (reproduce the paper results for the ADNI data set):**
```shell
python hyperparameters_optimization.py --input_data_file=../vader_data/ADNI/Xnorm.csv
                                       --input_data_type=ADNI
                                       --n_proc=6
                                       --n_repeats=20
                                       --n_sample=90
                                       --n_consensus=1
                                       --n_epoch=20
                                       --n_splits=2
                                       --n_perm=1000
                                       --output_folder=../vader_hp_opt_results_paper
```
**Detailed explanation of the script parameters:**

| Name          | Default                     | Typical range   | Description   |
| ------------- | --------------------------- | --------------- | ------------- |
| n_proc        | 6                           | 1-8             | Defines how many processor units can be used to run optimization jobs. If the value is too big - maximum number of CPUs will be used. Since each jobs splits into some sub-processes too, a good approach will be to set n_proc to a maximum number of CPUs divided by 4. |
| n_repeats     | 10                          | 10-20           | Defines how many times we perform the optimization for the same set of hyperparameters. The higher this parameter - the better is optimization, but the worse is performance. |
| n_sample      | None (full grid search)     | 30-150          | Defines how many sets of hyperparameters (excluding 'k'-s) we choose to evaluate from the full grid. For example, the full parameters grid described in the paper contains 896 sets of hyperparameters. If we set n_sample >= 896 or None, it will perform full grid search. If we set n_sample=100, it will randomly choose 100 sets of hyperparameters from the full grid. Note that if we test for 10 different k-s, the number of jobs will be multiplied. For example, if n_sample=100 and k is in range(2, 11), the total number of jobs will be 900. The higher this parameter - the better is optimization, but the worse is performance.  |
| n_consensus   | 1 (no consensus clustering) | 1-10            | Defines how many times we train vader for each job for each data split. If n_consensus > 1, then it runs the "consensus clustering" algorithm to determine the final clustering. The higher this parameter - the better is optimization, but the worse is performance.  |
| n_splits      | 2                           | 2-10            | Defines into how many chunks we split the data for the cross-validation step. Increase this parameter for bigger data sets.  |
| n_epoch       | 10                          | 10-50           | Defines how many epochs we train during the vader's "fit" step. The higher this parameter - the better is optimization, but the worse is performance.  |
| n_perm        | 100                         | 100-1000        | Defines how many times we permute each clustering during the calculation of the "prediction_strength_null". The higher this parameter - the better is optimization, but the worse is performance.  |
| seed          | None                        | Any integer     | Initializes the random number generator. It can be used to achieve reproducible results. If None - the random number generator will use its in-built initialization logic (e.g. using the current system time)  |
| output_folder | Current folder              | Any folder path | Defines a folder where all outputs will be written. Outputs include:<ul><li>final pdf report;</li><li>diffs csv file that was used to generate the pdf report;</li><li>all jobs results in csv format;</li><li>"csv_repeats" folder with intermediate csv chunks;</li><li>"failed_jobs" folder with stack-traces for all failed jobs;</li><li>logging file.</li></ul>  |

The processing time is proportional to ``n_sample * n_repeats * n_splits * n_consensus * n_epoch / n_proc``.

**Output report naming convention**

Generated reports have the following name structure:
```
adni_report_n_grid<n>_n_sample<n>_n_repeats<n>_n_splits<n>_n_consensus<n>_n_epoch<n>_n_perm<n>_seed<n>.pdf
```
The order of the parameters represents the sequence of processing. ``n_grid`` goes first, because we generated the parameter grid in the beginning of the process. Then, ``n_sample`` goes, because we picked up random samples right after we generated the parameters grid. Then, ``n_repeats`` goes, and so on.

#### VaDER CLI
Similar to the hyperparameters optimization, there must be a function that transforms input data into a tensor; How to integrate it - see the explanation in the part "Hyperparameters optimization (preparation)"

**Usage**
```shell
python run_vader.py [-h] --input_data_file INPUT_DATA_FILE [--input_weights_file INPUT_WEIGHTS_FILE] --input_data_type {ADNI,NACC,PPMI,custom} [--n_repeats N_REPEATS]
                    [--n_epoch N_EPOCH] [--n_consensus N_CONSENSUS] --k K --n_hidden N_HIDDEN [N_HIDDEN ...] --learning_rate LEARNING_RATE --batch_size BATCH_SIZE --alpha ALPHA
                    [--save_path SAVE_PATH] [--seed SEED] --report_file_path REPORT_FILE_PATH

optional arguments:
  -h, --help            show this help message and exit
  --input_data_file INPUT_DATA_FILE
                        a .csv file with input data
  --input_weights_file INPUT_WEIGHTS_FILE
                        a .csv file with flags for missing values
  --input_data_type {ADNI,NACC,PPMI,custom}
                        data type
  --n_repeats N_REPEATS
                        number of repeats
  --n_epoch N_EPOCH     number of training epochs
  --n_consensus N_CONSENSUS
                        number of repeats for consensus clustering
  --k K                 number of repeats
  --n_hidden N_HIDDEN [N_HIDDEN ...]
                        hidden layers
  --learning_rate LEARNING_RATE
                        learning rate
  --batch_size BATCH_SIZE
                        batch size
  --alpha ALPHA         alpha
  --save_path SAVE_PATH
                        model save path
  --seed SEED           seed
  --report_file_path REPORT_FILE_PATH
```
The script result will be represented as a txt report written in a given```report_file_path```.


**Example:**
```shell
python run_vader.py --input_data_file=../vader_data/NACC/Nacc.csv
                    --input_data_type=NACC
                    --n_repeats=1
                    --n_epoch=20
                    --n_consensus=5
                    --k=4
                    --n_hidden 32 8
                    --learning_rate=1e-2
                    --batch_size=64
                    --alpha=1
                    --report_file_path=nacc_report.txt
```
