# optim-benchmark: Benchmarking gradient-free optimisers on synthetic functions

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Open Source Love png2](https://badges.frapsoft.com/os/v2/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lazyoracle/optim-benchmark/master)

* [Overview](#overview)
* [Usage](#usage)
  * [Run Experiment](#run-experiment)
  * [Visualising Results](#visualising-results)
* [Optimisation Algorithms & Benchmark Functions](#optimisation-algorithms---benchmark-functions)
* [Results](#results)
  * [Easy](#easy)
  * [Hard](#hard)
  * [Number of Evaluations to a Goal](#number-of-evaluations-to-a-goal)

## Overview

This repository benchmarks various gradient-free (black-box) optimisers on synthetic functions. Specifically, we look at the following variables when considering standard benchmark functions:

* Noise
* Dimensions
* Effects of logarithms

## Usage

The code is designed to work with **Python 3.6.1+**. All required libraries are listed in `requirements.txt`, but the key ingredients are `nevergrad`, `cma`, `scikit-optimize` and `bayesian-optimization`. We recommend creating an environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
git clone https://github.com/lazyoracle/optim-benchmark
cd optim-benchmark
conda create --name=benchmark python==3.6.1
conda activate benchmark
pip install -r requirements.txt
jupyter lab --port 4242 demo_notebook.ipynb
```

### Run Experiment

There are two steps to running an experiment:

* Defining experiment conditions
* Invoking the code to parse conditions and run experiment

Here's an example:

```python
algo_list = ['CMA', 'NoisyBandit', 'NoisyOnePlusOne', 'PSO',
             'RandomSearch', 'SPSA', 'TBPSA']
func_list = ['rosenbrock', 'sphere4', 'rastrigin', 'griewank', 'deceptivepath']
dim_list = [2, 3, 5, 8]
eps_list = [0.5, 0.10, 0.05, 0.03, 0.02, 0.01, 0]
log_list = [False]
EVAL_BUDGET = 1000
#CREATE A NEW FILE IF CHANGING THE NUMBER OF EVALUATIONS
saved_file = "results-low-dim.pkl" #File to read from where previous expts were stored
                            # or new file to write to
save_interval = 600
initials = [0.0, 5.0, -5.0, 23.46, -23.46]

run_exp(algo_list, func_list, dim_list, eps_list, log_list, EVAL_BUDGET, saved_file, 'pkl', initials, save_interval)
```

### Visualising Results

Generating plots from the current experiment or previously run experiments is also straightforward.

```python
exp_df = pd.read_pickle('results-low-dim.pkl')

stripped_expt_df = exp_df.drop(columns = ['exp_data', 'min_params', 'f_min', 'time'])
results_summary(stripped_expt_df)

eval_budget = 1000
filter_func = lambda z: ((z['func'] in ['rastrigin']) and
                     (z['dim'] == 5) and
                     (z['log'] in [False]) and
                     (z['algo'] in ['CMA', 'NoisyBandit', 'NoisyOnePlusOne', 'PSO',
                                     'RandomSearch', 'SPSA', 'TBPSA']) and
                     (z['starter'] in [5.0]) and  
                     (z['noise_level'] in [0.03]))
use_tex = False
fig_test = plot_regular(exp_df, filter_func, use_tex, plot_evals = eval_budget, y_field = 'f_min', logplot='y')
```

## Optimisation Algorithms & Benchmark Functions

The algorithm and some of the benchmark function implementations are taken from `nevergrad`. While the code should work with all algorithms listed in `nevergrad`, we specifically look at the following here:

* CMA
* Random Search
* Noisy Bandit
* Powell’s
* SPSA
* Differential Evolution
* PSA
* (1+1)
* Particle Swarm
* Nelder Mead
* Bayesian Optimisation
* Estimation of Distribution

We test the above algorithms on the following synthetic functions:

* Translated Sphere
![Plot of Translated Sphere](img/Picture1.png)
* Rosenbrock
![Plot of Rosenbrock](img/Picture2.png)
* Ill-Conditioned
![Plot of Ill-Conditioned Function](img/Picture3.png)
* Multimodal
![Plot of Multi-Modal Function](img/Picture4.png)
* Path-Function
![Plot of Path Function](img/Picture5.png)

This is a visual representation of what the landscape of these functions typically look like:

## Results

We group the combined effects of function landscape, noise and dimensions into *Easy*, *Medium*, *Medium-Hard* and *Hard* and study the minimisation of the funtion as the algorithm progresses with more function evaluations. The code is also designed to study the number of evaluations that would be required to reach a specific `f_min` goal.

Some preliminary results can be seen below:

### Easy

> * Convex (translated sphere)
> * noise 0.03
> * dimension 3
> * X: log-scale evaluations
> * Y: log-scale goal

![Plot of Easy Optimisation Benchmark](img/Picture6.png)

### Hard

> * Ill-Conditioned
> * noise 0.1
> * dimension 1000
> * X: log-scale evaluations
> * Y: log-scale goal

![Plot of Hard Optimisation Benchmark](img/Picture7.png)

### Number of Evaluations to a Goal

> * Rosenbrock
> * noise 0.1
> * Goal 1e-7

![Plot of Number of Evaluations to a goal](img/Picture8.png)

More details on preliminary results can be seen in the [presentation from July 2019](benchmark_optimisers_anurag.pdf)
