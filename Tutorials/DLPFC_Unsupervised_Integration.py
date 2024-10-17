import os
import anndata
import scanpy as sc
import random
import torch
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import silhouette_score as s_score

import stClinic as stClinic

warnings.filterwarnings("ignore")

os.environ['R_HOME'] = '/sibcb/program/install/r-4.0/lib64/R'
os.environ['R_USER'] = '/sibcb1/chenluonanlab8/zuochunman/anaconda3/envs/stClinic-test/lib/python3.8/site-packages/rpy2'

used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(used_device)

parser  =  stClinic.parameter_setting()
args    =  parser.parse_args()

args.input_dir = '/sibcb1/chenluonanlab8/zuochunman/Share_data/xiajunjie/Datasets/DLPFC/'

args.out_dir   = '/sibcb1/chenluonanlab8/cmzuo/workPath/Software/stClinic/stClinic_out/'
Path(args.out_dir).mkdir(parents=True, exist_ok=True)


# Set seed
seed    = 666
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Load data
section_ids = ['151673','151674','151675','151676']
print(section_ids)
Batch_list  = []
adj_list    = []

for idx, section_id in enumerate(section_ids):

	# Read h5 file
	input_dir = os.path.join(args.input_dir, section_id)
	adata     = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5', load_images=True)
	adata.var_names_make_unique(join="++")

	# Read corresponding annotation file
	Ann_df = pd.read_csv(os.path.join(input_dir, section_id + '_annotation.txt'), sep='\t', header=0, index_col=0)
	Ann_df.loc[Ann_df['Layer'].isna(),'Layer'] = "unknown"
	adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Layer'].astype('category')
	adata.obs['batch_name_idx'] = idx

	# Make spot name unique
	adata.obs_names = [x+'_'+section_id for x in adata.obs_names]

	# Construct intra-edges
	stClinic.Cal_Spatial_Net(adata, rad_cutoff=args.rad_cutoff)

	# Normalization
	sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.n_top_genes)
	sc.pp.normalize_total(adata, target_sum=1e4)
	sc.pp.log1p(adata)
	adata = adata[:, adata.var['highly_variable']]

	sc.tl.pca(adata, n_comps=10, random_state=seed)

	adj_list.append(adata.uns['adj'])
	Batch_list.append(adata)

# Concat scanpy objects
adata_concat = anndata.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs['Ground Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"]   = adata_concat.obs["slice_name"].astype('category')
print('\nShape of concatenated AnnData object: ', adata_concat.shape)

# Construct unified graph
# mnn_dict = create_dictionary_mnn(adata_concat, use_rep='X_pca', batch_name='batch_name', k=1) # k=0
adj_concat = stClinic.inter_linked_graph(adj_list, section_ids, mnn_dict=None)
adata_concat.uns['adj']      = adj_concat
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

# Run stClinic for unsupervised integration
centroids_num = args.n_centroids
print(f'Estimated centroids number: {centroids_num}')
adata_concat  = stClinic.train_stClinic_model(adata_concat, n_centroids=centroids_num, lr=args.lr_integration, device=used_device)

# Clustering
stClinic.mclust_R(adata_concat, num_cluster=len(np.unique(adata_concat.obs[adata_concat.obs['Ground Truth']!='unknown']['Ground Truth'])), used_obsm='stClinic')
adata_concat = adata_concat[adata_concat.obs['Ground Truth']!='unknown']

Batch_list1   = []
for section_id in section_ids:
	Batch_list1.append(adata_concat[adata_concat.obs['batch_name'] == section_id])

# UMAP reduction
sc.pp.neighbors(adata_concat, use_rep='stClinic', random_state=seed)
sc.tl.umap(adata_concat, random_state=seed)
spot_size  = 200
title_size = 12
ARI_list   = []
for bb in range(4):
	ARI_list.append(round(ari_score(Batch_list1[bb].obs['Ground Truth'], Batch_list1[bb].obs['mclust']), 2))

colors  = ["#6D1A9B", "#CB79A6", "#7494D2", "#59BD85", "#56B3E8", "#FDB815", "#F46867"]
fig, ax = plt.subplots(1, 4, figsize=(10, 5), gridspec_kw={'wspace': 0.05, 'hspace': 0.1})

_sc_0 = sc.pl.spatial(Batch_list1[0], img_key=None, color=['mclust'], title=[''],spot_size=200, legend_loc=None, 
					  legend_fontsize=12, show=False, ax=ax[0], frameon=False, palette=colors)
_sc_0[0].set_title("ARI=" + str(ARI_list[0]), size=12)
_sc_1 = sc.pl.spatial(Batch_list1[1], img_key=None, color=['mclust'], title=[''],spot_size=200, legend_loc=None,
					  legend_fontsize=12, show=False, ax=ax[1], frameon=False, palette=colors)
_sc_1[0].set_title("ARI=" + str(ARI_list[1]), size=12)
_sc_2 = sc.pl.spatial(Batch_list1[2], img_key=None, color=['mclust'], title=[''],spot_size=200, legend_loc=None,
					  legend_fontsize=12, show=False, ax=ax[2], frameon=False, palette=colors)
_sc_2[0].set_title("ARI=" + str(ARI_list[2]), size=12)
_sc_3 = sc.pl.spatial(Batch_list1[3], img_key=None, color=['mclust'], title=[''],spot_size=200, legend_loc=None,
					  legend_fontsize=12, show=False, ax=ax[3], frameon=False, palette=colors)
_sc_3[0].set_title("ARI=" + str(ARI_list[3]), size=12)

plt.savefig(args.out_dir + 'DLPFC_intergrated_{}.png'.format(section_str))

# Save AnnData object    (only X, obs & obsm)
del adata_concat.uns; del adata_concat.obsp

section_str = '_'.join(section_ids)
adata_concat.write(args.out_dir + 'integrated_adata_{}.h5ad'.format(section_str), compression='gzip')
