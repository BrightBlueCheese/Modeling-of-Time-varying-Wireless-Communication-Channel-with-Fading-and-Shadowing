# Modeling-of-Time-varying-Wireless-Communication-Channel-with-Fading-and-Shadowing 

<!-- This repository is the official implementation of [Modeling of Time-varying Wireless Communication Channel with Fading and Shadowing](https://arxiv.org/abs/2405.00949).  -->
This repository is the official implementation of [Modeling of Time-varying Wireless Communication Channel with Fading and Shadowing](Not yet uploaded). 

## Requirements

To install requirements:

```setup
pip install -r requirements_cuda118.txt
```

>📋  The experiments were done under CUDA 11.8

## Dataset

1. Data : ```data.py```
2. Data Hyper-paremeters : ```data_hp.py```

>📋  You don't have to run those files

## Training
1. Train Nakagami 1 from Random Init : ```python run_n.py```
2. Train Nakagami 1 with N_G_var(10x-16) from Random Init : ```python run_ne2.py```
3. Train Log-Normal 1 from Random Init : ```python run_ln.py```
Then you have to open and work through './DataAnalysis/0_Nakagami1_Eval.ipynb' and './DataAnalysis/1_LogNormal1_Eval.ipynb' to select the Global Best (and Median) Nakagami 1 (Log-Normal 1).

The git repository already have `Nakagami_XXX.h5` and `LogNormal_XXX.h5`, but those are from the research paper.

You need to replaced those with yours if you want to try with your owns.
4. Train Nakagami 2 from Random Init : ```python run_n2.py```
5. Train Nakagami 2 from Global Best (and Median) Nakagami 1 : ```python run_n1_n2.py```. Please check the file before run.
6. Train Log-Normal 2 from Random Init : ```python run_ln2.py```
7. Train Log-Normal 2 from Global Best (and Median) Nakagami 2 : ```python run_ln1_ln2.py```. Please check the file before run.

>📋  You can control the DMDN model's and its training hyper-parameters with `model.py` and `model_hp.py`

## Evaluation

Move to DataAnalysis

1. Detail information and the visualizations of the data : ```Data Visualization.ipynb```
2. Evaluate and select the Global Best (and Median) Nakagami 1 : ```0_Nakagami1_Eval.ipynb```
2. Evaluate and select the Global Best (and Median) Log-Normal 1 : ```1_LogNormal1_Eval.ipynb```
4. Evaluate Nakagami 2 cases : ```2_Nakagami1_Nakagami2_Eval.ipynb```
5. Evaluate Log-Normal 2 cases : ```3_LogNormal1_LogNormal2_Eval.ipynb```

## Contributing

>📋  MIT

## Authors' Note
Please use this code only for social goods and positive impact.
