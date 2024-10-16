# stClinic

*stClinic dissects clinically relevant niches by integrating spatial multi-slice multi-omics data in dynamic graphs.*

## Overview

![image](https://github.com/cmzuo11/stClinic/blob/main/image/stClinic_logo.png)

stClinic can integrate spatial multi-slice omics data from the same tissue or different tissues, and spatial multi-omics data from the same slice or different slices/technologies.

## Installation

The installation was tested on a machine with a 40-core Intel(R) Xeon(R) Silver 4210R CPU, 128GB of RAM, and an NVIDIA A800 GPU with 80GB of RAM, using Python 3.8.17, Scanpy 1.9.3, PyTorch 1.12.0, and PyG (PyTorch Geometric) 2.3.0. If possible, please run stClinic on CUDA.

### 1. Grab source code of stClinic

```bash
git clone https://github.com/cmzuo11/stClinic.git
cd stClinic-main
```

### 2. Install stClinic in the virtual environment by conda

* First, install conda: https://docs.anaconda.com/anaconda/install/index.html
* Then, all packages used by stClinic (described by "environment.yml") are automatically installed in a few minutes.

```bash
conda config --set channel_priority strict
conda env create -f environment.yml
conda activate stClinic
```

The specific software dependencies are listed in "requirements.txt".

## Tutorials

* [Tutorial 1](https://github.com/cmzuo11/stClinic/wiki/Tutorial-1:-10X-Visium-(DLPFC-dataset)): Integrating slices 151673, 151674, 151675, and 151676 of the DLPFC dataset.
* [Tutorial 2](https://github.com/cmzuo11/stClinic/wiki/Tutorial-2:-Integration-Analysis-of-Primary-Colorectal-Cancer-and-Liver-Metastasis-Slices): Integrating 24 slices comprising 14 primary colorectal cancers and 10 liver metastases.
* [Tutorial 3](https://github.com/cmzuo11/stClinic/wiki/Tutorial-3:-Supervised-Model-for-Identifying-Metastasis%E2%80%90Related-Niches-Using-Primary-Colorectal-Cancer-and-Liver-Metastasis-Slices): Identifying metastasis-related TMEs through integrative analysis of primary colorectal cancer and liver metastasis slices.

## References

* stMVC: https://github.com/cmzuo11/stMVC
* STAligner: https://github.com/zhoux85/STAligner
* CellCharter: https://github.com/CSOgroup/cellcharter
* CytoCommunity: https://github.com/huBioinfo/CytoCommunity
* PathFinder: https://github.com/Biooptics2021/PathFinder

