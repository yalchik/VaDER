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

#### 1. Data reader
There must be a class that transforms input data into a tensor. It should be declared in a separate file and follow the following convention:
* Class name is `DataReader`.
* It implements abstract class `vader.hp_opt.interface.abstract_data_reader.AbstractDataReader`.
* It has no required init parameters.

You can find examples in the `tools/addons/data_reader` folder.

Out of the box, there are the following data readers:
1. Clean normalized ADNI data: `tools/addons/data_reader/adni_norm_data.py`.
2. Raw non-normalized ADNI data: `tools/addons/data_reader/adni_raw_data.py`.
3. Clean non-normalized NACC data: `tools/addons/data_reader/nacc_data.py`.
4. Raw non-normalized NACC data: `tools/addons/data_reader/nacc_raw_data.py`.

How to use it? - Pass the script path to the `--data_reader_script` input parameter. Example:
```shell
python hyperparameters_optimization.py --data_reader_script=tools/addons/data_reader/adni_norm_data.py ...
```

#### 2. Parameters factory.
There must be a class that declares ranges of possible values for hyperparameters. It should be declared in a separate file and follow the following conventions:
* Class name is `ParamsFactory`.
* It implements one of 2 abstract classes: `vader.hp_opt.interface.abstract_grid_search_params_factory.AbstractGridSearchParamsFactory` or `from vader.hp_opt.interface.abstract_bayesian_params_factory.AbstractBayesianParamsFactory` (depending on which type of hyperparameters optimization you'd like to use - grid search or bayesian).
* It has no required init parameters.

Out-of-box, there are the following parameters factories:
1. Lightweight grid search parameters factory: `tools/addons/params_factory/grid_search_params.py`
2. Broad grid search parameters factory (repeats the vader paper experiment): `tools/addons/params_factory/grid_search_paper_params.py`
3. Bayesian parameters factory: `tools/addons/params_factory/bayesian_params.py`

How to use it? - Pass the script path to the `--param_factory_script` input parameter. Example:
```shell
python hyperparameters_optimization.py --param_factory_script=tools/addons/params_factory/grid_search_params.py  ...
```

#### Hyperparameters optimization CLI
**Usage**
```shell
python hyperparameters_optimization.py [-h] --input_data_file INPUT_DATA_FILE [--input_weights_file INPUT_WEIGHTS_FILE] [--input_seed INPUT_SEED]
                                       [--param_factory_script PARAM_FACTORY_SCRIPT] [--data_reader_script DATA_READER_SCRIPT] [--n_repeats N_REPEATS] [--n_proc N_PROC]
                                       [--n_sample N_SAMPLE] [--n_consensus N_CONSENSUS] [--n_epoch N_EPOCH] [--early_stopping_ratio EARLY_STOPPING_RATIO]
                                       [--early_stopping_batch_size EARLY_STOPPING_BATCH_SIZE] [--n_splits N_SPLITS] [--n_perm N_PERM] [--type {gridsearch,bayesian}]
                                       [--n_trials N_TRIALS] --output_folder OUTPUT_FOLDER [--enable_cv_loss_reports]

optional arguments:
  -h, --help            show this help message and exit
  --input_data_file INPUT_DATA_FILE
                        .csv file with input data
  --input_weights_file INPUT_WEIGHTS_FILE
                        .csv file with flags for missing values
  --input_seed INPUT_SEED
                        used both as KFold random_state and VaDER seed
  --param_factory_script PARAM_FACTORY_SCRIPT
                        python script declaring param grid factory
  --data_reader_script DATA_READER_SCRIPT
                        python script declaring data reader class
  --n_repeats N_REPEATS
                        number of repeats, default 10
  --n_proc N_PROC       number of processor units that can be used, default 6
  --n_sample N_SAMPLE   number of hyperparameters set per CV, default - full grid
  --n_consensus N_CONSENSUS
                        number of repeats for consensus clustering, default 1
  --n_epoch N_EPOCH     number of epochs for VaDER training, default 10
  --early_stopping_ratio EARLY_STOPPING_RATIO
                        early stopping ratio
  --early_stopping_batch_size EARLY_STOPPING_BATCH_SIZE
                        early stopping batch size
  --n_splits N_SPLITS   number of splits in KFold per optimization job, default 2
  --n_perm N_PERM       number of permutations for prediction strength, default 100
  --type {gridsearch,bayesian}
  --n_trials N_TRIALS   number of trials (for bayesian optimization only), default 100
  --output_folder OUTPUT_FOLDER
                        a directory where report will be written
  --enable_cv_loss_reports
```
The script result will be represented as a PDF report written in a given```output_folder```.

**Example (local smoke test):**
```shell
python hyperparameters_optimization.py --input_data_file=../data/ADNI/Xnorm.csv
                                       --param_factory_script=addons/params_factory/grid_search_params.py
                                       --data_reader_script=addons/data_reader/adni_norm_data.py
                                       --n_proc=5
                                       --n_repeats=5
                                       --n_sample=5
                                       --n_consensus=1
                                       --n_epoch=10
                                       --n_splits=2
                                       --n_perm=10
                                       --output_folder=../vader_results 
```

**Example (reproduce the paper results for the ADNI data set):**
```shell
python hyperparameters_optimization.py --input_data_file=../vader_data/ADNI/Xnorm.csv
                                       --param_factory_script=addons/params_factory/grid_search_paper_params.py
                                       --data_reader_script=addons/data_reader/adni_norm_data.py
                                       --n_proc=6
                                       --n_repeats=20
                                       --n_sample=90
                                       --n_consensus=1
                                       --n_epoch=50
                                       --n_splits=2
                                       --n_perm=1000
                                       --output_folder=../vader_hp_opt_results_paper
```
**Detailed explanation of the script parameters:**


| Name          | Default                     | Typical range   | Description   |
| ------------- | --------------------------- | --------------- | ------------- |     
| type          | gridsearch                 | <ul><li>gridsearch</li><li>bayesian</li></ul> | Defines which type of hyperparameters optimization we run. Grid search is better for parallelization, while the bayesian optimization can find better sets of hyperparameters. |
| n_proc        | 6                           | 1-8             | Defines how many processor units can be used to run optimization jobs. If the value is too big - maximum number of CPUs will be used. Since each jobs splits into some sub-processes too, a good approach will be to set n_proc to a maximum number of CPUs divided by 4. |
| n_repeats     | 10                          | 10-20           | Defines how many times we perform the optimization for the same set of hyperparameters. The higher this parameter - the better is optimization, but the worse is performance. |
| n_sample      | None (full grid search)     | 30-150          | Defines how many sets of hyperparameters (excluding 'k'-s) we choose to evaluate from the full grid. For example, the full parameters grid described in the paper contains 896 sets of hyperparameters. If we set n_sample >= 896 or None, it will perform full grid search. If we set n_sample=100, it will randomly choose 100 sets of hyperparameters from the full grid. Note that if we test for 10 different k-s, the number of jobs will be multiplied. For example, if n_sample=100 and k is in range(2, 11), the total number of jobs will be 900. The higher this parameter - the better is optimization, but the worse is performance. This parameters work only for grid search (`--type=grid_search`). For bayesian optimization, this parameters does not have any effect.  |
| n_trials      | 100                         | 100-300         | Defines how many sets of hyperparameters (excluding 'k'-s) we choose to evaluate. Each set of hyperparameters is chosen automatically according to the Bayesian optimization rules based on the performance of previous hyperparameters. The higher this parameter - the better is optimization, but the worse is performance. It works only for bayesian optimization (`--type=bayesian`). For grid search, it does not have any effect. |
| n_consensus   | 1 (no consensus clustering) | 1-10            | Defines how many times we train vader for each job for each data split. If n_consensus > 1, then it runs the "consensus clustering" algorithm to determine the final clustering. The higher this parameter - the better is optimization, but the worse is performance.  |
| n_splits      | 2                           | 2-10            | Defines into how many chunks we split the data for the cross-validation step. Increase this parameter for bigger data sets.  |
| n_perm        | 100                         | 100-1000        | Defines how many times we permute each clustering during the calculation of the "prediction_strength_null". The higher this parameter - the better is optimization, but the worse is performance.  |
| n_epoch       | 10                          | 10-50           | Defines how many epochs we train during the vader's "fit" step. The higher this parameter - the better is optimization, but the worse is performance.  |
| early_stopping_ratio | None                 | 0.01-0.1        | Defines the relative difference at which the model can stop fitting. Optimal value: 0.03 (which means that we stop fitting the model once loss changes less than 3% on average). |
| early_stopping_batch_size | 5               | 5-10            | Defines how many epochs we use to calculate average relative difference in loss for early stopping criteria. When early_stopping_ratio is None, it does not have any effect. |
| seed          | None                        | Any integer     | Initializes the random number generator. It can be used to achieve reproducible results. If None - the random number generator will use its in-built initialization logic (e.g. using the current system time)  |
| output_folder | Current folder              | Any folder path | Defines a folder where all outputs will be written. Outputs include:<ul><li>final pdf report;</li><li>diffs csv file that was used to generate the pdf report;</li><li>all jobs results in csv format;</li><li>"csv_repeats" folder with intermediate csv chunks;</li><li>"failed_jobs" folder with stack-traces for all failed jobs;</li><li>logging file.</li></ul>  |
| enable_cv_loss_reports | False              | True or False   | If true, the program will produce intermittent reports showing loss changes over epochs during cross-validation. |

The processing time is proportional to ``n_sample * n_repeats * n_splits * n_consensus * n_epoch / n_proc``.

**Output report naming convention**

Generated reports have the following name structure:
```
adni_report_n_grid<n>_n_sample<n>_n_repeats<n>_n_splits<n>_n_consensus<n>_n_epoch<n>_n_perm<n>_seed<n>.pdf
```
The order of the parameters represents the sequence of processing. ``n_grid`` goes first, because we generated the parameter grid in the beginning of the process. Then, ``n_sample`` goes, because we picked up random samples right after we generated the parameters grid. Then, ``n_repeats`` goes, and so on.

#### VaDER CLI
Similar to the hyperparameters optimization, there must be a DataReader class; How to integrate it - see the explanation in the part "Hyperparameters optimization (preparation)"

**Usage**
```shell
python run_vader.py [-h] --input_data_file INPUT_DATA_FILE [--input_weights_file INPUT_WEIGHTS_FILE] [--data_reader_script DATA_READER_SCRIPT] [--n_epoch N_EPOCH]
                    [--early_stopping_ratio EARLY_STOPPING_RATIO] [--early_stopping_batch_size EARLY_STOPPING_BATCH_SIZE] [--n_consensus N_CONSENSUS] --k K --n_hidden N_HIDDEN
                    [N_HIDDEN ...] --learning_rate LEARNING_RATE --batch_size BATCH_SIZE --alpha ALPHA [--save_path SAVE_PATH] [--seed SEED] --output_path OUTPUT_PATH

optional arguments:
  -h, --help            show this help message and exit
  --input_data_file INPUT_DATA_FILE
                        a .csv file with input data
  --input_weights_file INPUT_WEIGHTS_FILE
                        a .csv file with flags for missing values
  --data_reader_script DATA_READER_SCRIPT
                        python script declaring data reader class
  --n_epoch N_EPOCH     number of training epochs
  --early_stopping_ratio EARLY_STOPPING_RATIO
                        early stopping ratio
  --early_stopping_batch_size EARLY_STOPPING_BATCH_SIZE
                        early stopping batch size
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
  --output_path OUTPUT_PATH
```
The script result will be represented as a txt report written in a given```output_path```.


**Example:**
```shell
python run_vader.py --input_data_file=../vader_data/ADNI/Xnorm.csv
                    --data_reader_script=tools/addons/data_reader/adni_norm_data.py
                    --n_epoch=50
                    --n_consensus=20
                    --k=3
                    --n_hidden 102 1
                    --learning_rate=0.000265
                    --batch_size=75
                    --alpha=1
                    --output_path=../vader_results/ADNI_02_25_run_1
```

### Docker integration
Docker image is located here: https://hub.docker.com/repository/docker/yalchik/vader

It requires setting 2 environment variables:
1. `SCRIPT` defines which python script from the `tools` folder you'd like to run. Possible values: `hyperparameters_optimization` and `run_vader`.
2. `SCRIPT_ARGS` defines everything you'd like to pass as input arguments to that script.

Example:
```shell
docker run --rm --name vader
  -v /home/iyalchyk/vader_data/ADNI/Xnorm.csv:/usr/input/Xnorm.csv
  -v /home/iyalchyk/vader_result:/usr/output
  -e SCRIPT=run_vader
  -e SCRIPT_ARGS="--input_data_file=/usr/input/Xnorm.csv --data_reader_script=/usr/src/app/tools/addons/data_reader/adni_norm_data.py --n_epoch=50 --n_consensus=20 --k=3 --n_hidden 47 10 --learning_rate=0.0006 --batch_size=96 --alpha=1 --output_path=/usr/output"
  vader
```
