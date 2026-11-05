## Installation

### Create a virtual environment

We recommend users create a virtual environment using Conda. Users can install `anaconda` or `miniconda` according to the guide at [https://www.anaconda.com/].

Create a virtual environment by running:

```shell
conda create -n diffgsp_env python=3.9.0
conda activate diffgsp_env
```

### Download DiffGSP

Run:

```shell
git clone https://github.com/jxLiu-bio/DiffGSP
cd ./DiffGSP
```

### Install DiffGSP

After completing the above procedures, install DiffGSP by running:

```shell
pip install .
```

Test whether the installation was successful by launching Python:

```shell
python
```

Then run:

```python
import DiffGSP as dg
```

### Install PyTorch

DiffGSP supports GPU acceleration through PyTorch.

For CUDA 12.4, install PyTorch and Torchvision using:

```shell
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only environments, use:

```shell
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

For optional GPU acceleration using CuPy with CUDA 12.x, install:

```shell
pip install cupy-cuda12x==13.6.0
```

Users can select the appropriate PyTorch version according to their CUDA environment. For other configurations, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Install Jupyter (Optional)

We recommend [Jupyter](https://jupyter.org/) for interactive usage. It can be installed and configured by running:

```shell
pip install jupyter ipykernel
python -m ipykernel install --user --name=diffgsp_env --display-name=diffgsp_env
```