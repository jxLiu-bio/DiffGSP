# DiffGSP: Modeling and Reversing mRNA Diffusion in Spatial Transcriptomics Using Fick’s Law and Graph Signal Processing
<img src="https://img.shields.io/badge/Platform-Linux-green"> <img src="https://img.shields.io/badge/Language-python3-green"> <img src="https://img.shields.io/badge/License-MIT-green"><img src="https://img.shields.io/badge/notebooks-passing-green"><img src="https://img.shields.io/badge/docs-passing-green">

Spatial transcriptomics enables the measurement of gene expression within tissues while preserving spatial context, offering deeper insights into tissue organization and intercellular interactions. However, transcripts captured at a given spot may originate from multiple adjacent regions due to mRNA diffusion during the data generation process, compromising data reliability and leading to misinterpretation of downstream analyses. To address the data inaccuracy caused by molecular diffusion in sequencing-based spatial transcriptomics, we present DiffGSP, a first-principles-based method that applies Fick's law in combination with graph signal processing to mitigate molecular diffusion effects and denoise spatial transcriptomics data, thereby recovering true gene expression profiles.


## System Requirments
### OS requirements
DiffGSP is based on Python Platform and can be run on Windows, Linux. The package has been tested on 
the following systems:

- Linux (recommend): CentOS 7.9.2009, Ubuntu 18.04, Ubuntu 20.04 
- Windows: Windows 10

### Python Dependencies
DiffGSP requires the version of Python >= 3.8.


```
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.2
tqdm==4.66.4
matplotlib==3.7.5
scipy==1.10.1
scikit-learn==1.3.2
leidenalg==0.10.2
ipython==8.12.3
kneed==0.8.5
scanpy==1.9.8

```

## Installation Guide

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

## Usage and Tutorials
The detail step-by-step usage guide can be found in <https://diffgsp.readthedocs.io/en/latest/>





