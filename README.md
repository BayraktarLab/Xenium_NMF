# Xenium_NMF
Non-negative factorization for xenium data in pyro.

## Installation

Create a conda environment and install the `Xenium_NMF` package.

```bash
conda create -y -n Xenium_NMF python=3.9

conda activate Xenium_NMF
pip install git+https://github.com/BayraktarLab/Xenium_NMF.git@flexibility
```

To use this environment in a jupyter notebook, add a jupyter kernel for this environment:

```bash
conda activate Xenium_NMF
pip install ipykernel
python -m ipykernel install --user --name=Xenium_NMF --display-name='Environment (Xenium_NMF)'
```

## Tips

### Conda environment for A100 GPUs

```bash
conda create -y -n Xenium_NMF python=3.9

conda activate Xenium_NMF
pip install git+https://github.com/BayraktarLab/Xenium_NMF.git@flexibility
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```