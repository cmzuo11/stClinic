# stClinic

*stClinic dissects clinically relevant niches by integrating spatial multi-slice multi-omics data in dynamic graphs.*

## Overview

![image](https://github.com/JunjieXia14/stClinic/blob/main/image/stClinic_logo.png)

stClinic can integrate spatial multi-slice omics data from the same tissue or different tissues, and spatial multi-omics data from the same slice or different slices/technologies.

## Installation

The installation was tested on a machine with a 40-core Intel(R) Xeon(R) Silver 4210R CPU, 128GB of RAM, and an NVIDIA A800 GPU with 80GB of RAM, using Python 3.8.17, Scanpy 1.9.3, PyTorch 1.12.0, and PyG (PyTorch Geometric) 2.3.0. If possible, please run stClinic on CUDA.

### 1. Grab source code of stClinic

```bash
git clone https://github.com/JunjieXia14/stClinic.git
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

## Quick start

### Learning batch-corrected features across slices using unsupervised stClinic

#### Input

We use the human dorsolateral prefrontal cortex (DLPFC) dataset from 10X Visium (four slices: 151673-151676) generated by Maynard et al. (Nature Neuroscience, 2021) as an example input. This dataset includes three types of files: (1) gene expression data, (2) spatial location data, and (3) annotation labels for each spot. These input files are available in Datasets/README.md.

Note that you should first define the installation path for the R software as well as the rpy2 package.

#### Run

Run the following commands in Linux Bash Shell:

```bash
cd Tutorials/code
python DLPFC_Unsupervised_Integration.py --input_dir ../../Datasets/DLPFC
```

The script automatically (1) loads the input data as concatenated `AnnData` object, (2) builds an initial unified graph based on spatial locations and omics profiles, (3) learns batch-corrected features of four slices by stClinic in an unsupervised manner, (4) identifies spatial domains based on batch-corrected features using the `mclust` algorithm, and (5) maps the latent features into 2D-UMAP space. It takes 5 mins.

**Hyperparameters**

* rad_cutoff: The radius value used to construct intra-edge (i.e., spatial nearest neighbors) for each slice.
* k_cutoff: The number of spatial neighbors used to construct intra-edge. The default value is 6.
* k: The number of mutual nearest neighbors used to construct inter-edges across slices. In heterogeneous tissues such as tumors, the value is set to 5; in spatial multi-omics dataset, the value is set to 10; and in relatively homogeneous tissues, the value is 1 or 0.
* n_top_genes: The number of highly variable genes selected for each slice. The default value is 5000, which can be larger than the default for heterogeneous datasets.
* n_centroids: The number of components of the GMM.
* lr_integration: The learning rate used by stClinic when extracting batch-corrected features in slices. The default value is 0.0005. You can adjust it from 0.0005/20 to 0.0005 based on your data.

#### Output

An concatenated `AnnData` object with stClinic embeddings and UMAP coordinates of four slices stored in `AnnData.obsm`, and spatial cluster labels stored in `AnnData.obs.mclust`.

Note: To reduce your waiting time, we have uploaded the processed `AnnData` object into `Datasets/DLPFC`.

### Evaluating cluster importance using attention-based supervised learning

#### Input

We use the human colorectal cancer and liver metastasis (CRCLM) dataset from 10X Visium (24 slices) generated by Villemin et al. (Nucleic Acids Research, 2023), Valdeolivas et al. (npj Precision Oncology, 2024), Wu et al. (Cancer Discovery, 2022), Garbarino et al. (Aging Cell, 2023), and Wang et al. (Science Advances, 2023) as an example input for the supervised mode of stClinic. This dataset includes three types of files: (1) gene expression data, (2) spatial location data, and (3) sample labels. These input files are available in Datasets/README.md.

#### Run

We first run the following commands in Linux Bash Shell to (1) learn shared features for the 24 CRCLM slices, (2) identify spatial domains based on the latent features using the `Louvain` algorithm, and (3) project the latent features into 2D-UMAP space. It takes 20 mins.

```bash
cd Tutorials/code
python CRCLM_Unsupervised_Integration.py --input_dir ../../Datasets/CRCLM
```

Note: To reduce your waiting time, we have uploaded the processed `AnnData` object into `Datasets/CRCLM`.

We then run the following command in Linux Bash Shell:

```bash
python CRCLM_Supervised_Prediction.py
```

This script automatically (1) computes 6 statistics measures of each cluster, (2) fuses them into slice representations by attention mechanism, and (3) evaluates the importance level of TME in predicting clinical outcome. It takes 1 minute.

**Hyperparameters**

* The type of clinical data. The value is 'survival' when using survival time, and the value is 'grading' when using categorical data.
* lr_prediction: The learning rate used by supervised stClinic. The default value is 0.05. You can adjust it to the value with the highest C-Index or classification accuracy in the grid search program of cross-validation based on your data.

#### Output

An AnnData object encompasses AnnData.obsm, which holds the latent features of stClinic and the UMAP embeddings of latent features for the 24 slices; AnnData.obs.louvain, which comprises spatial clusters; and AnnData.uns, which stores the weights of each cluster.

## More tutorials

More detailed tutorials for each dataset are described in the `Tutorials/notebook` folder:

* Tutorial 1: Integrating slices 151673, 151674, 151675, and 151676 of the DLPFC dataset.
* Tutorial 2: Integrating two breast cancer slices.
* Tutorial 3: Identifying metastasis-related TMEs through integrative analysis of primary colorectal cancer and liver metastasis slices.

## References

* stMVC: https://github.com/cmzuo11/stMVC
* STAligner: https://github.com/zhoux85/STAligner
* CellCharter: https://github.com/CSOgroup/cellcharter
* CytoCommunity: https://github.com/huBioinfo/CytoCommunity
* PathFinder: https://github.com/Biooptics2021/PathFinder
