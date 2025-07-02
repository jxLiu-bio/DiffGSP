## Installation

### Create a virtual environment
We recommend Users to create a virtual environment by conda. Users can
install ```anaconda``` or ```miniconda``` according to the guide in [https://www.anaconda.com/].

Create a vritual environment by run

```shell
conda create -n diffgsp_env python==3.8
conda activate diffgsp_env
```

### DownLoad DiffGSP
Run
```shell
git clone https://github.com/jxLiu-bio/DiffGSP
cd ./DiffGSP

```

### Install DiffGSP
After above precudures, you can install DiffGSP by
```shell
pip install .
```

Test whether it has been install successfully by run following codes in Python
```shell
python
``` 
and 
```Python
import DiffGSP as dg
```
### Install Torch

Install the torch by
```shell
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
where cu116 can also be replaced to your cuda version or just cpu version.

### Install jupyter (optional)
Note that we recommend [jupyter](https://jupyter.org/) for interactive usage. It can be installed and configured by

```shell
conda install jupyter
python -m ipykernel install --user --name=diffgsp_env --display-name=diffgsp_env
```
