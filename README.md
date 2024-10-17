# stClinic
![image](https://github.com/cmzuo11/stClinic/blob/main/image/stClinic_logo.png)

stClinic can integrate spatial multi-slice omics data from the same tissue or different tissues, and spatial multi-omics data from the same slice or different slices/technologies.

## Installation

The installation was tested on a machine with a 40-core Intel(R) Xeon(R) Silver 4210R CPU, 128GB of RAM, and an NVIDIA A800 GPU with 80GB of RAM, using Python 3.8.17, Scanpy 1.9.3, PyTorch 1.12.0, and PyG (PyTorch Geometric) 2.3.0. If possible, please run stClinic on CUDA.

### Step1. Install stClinic in the virtual environment by conda

* First, install conda: https://docs.anaconda.com/anaconda/install/index.html
* Then, create a envs named stClinic with python 3.8.17

```bash
conda create -n stClinic python=3.8.17 pip
conda activate stClinic
```

### Step2. automatically install stClinic from pypi package
* If your CUDA version is in the list(https://data.pyg.org/whl/), please install the corresponding version torch pyg dependencies. We provide an exmaple for CUDA 12.4:

```bash
pip install stClinic
pip install pyg_lib torch-scatter torch-geometric torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html --no-cache-dir
```

## Tutorials

* [Tutorial 1](https://github.com/cmzuo11/stClinic/wiki/Tutorial-1:-10X-Visium-(DLPFC-dataset)): Integrating slices 151673, 151674, 151675, and 151676 of the DLPFC dataset.
* [Tutorial 2](https://github.com/cmzuo11/stClinic/wiki/Tutorial-2:-Integration-Analysis-of-Primary-Colorectal-Cancer-and-Liver-Metastasis-Slices): Integrating 24 slices comprising 14 primary colorectal cancers and 10 liver metastases.
* [Tutorial 3](https://github.com/cmzuo11/stClinic/wiki/Tutorial-3:-Supervised-Model-for-Identifying-Metastasis%E2%80%90Related-Niches-Using-Primary-Colorectal-Cancer-and-Liver-Metastasis-Slices): Identifying metastasis-related TMEs through integrative analysis of primary colorectal cancer and liver metastasis slices.
* [Quick start](https://github.com/cmzuo11/stClinic/wiki/Quick-start): Quick start for unsupervised and supervised stClinic.

