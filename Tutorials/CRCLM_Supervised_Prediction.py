import os
import scanpy as sc
import random
import torch
import numpy as np
import pandas as pd
import warnings
import re
from pathlib import Path

import stClinic as stClinic

warnings.filterwarnings("ignore")

used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser      =  stClinic.parameter_setting()
args        =  parser.parse_args()

args.input_dir = '/sibcb1/chenluonanlab8/cmzuo/workPath/Software/stClinic/stClinic_out/'
args.out_dir   = args.input_dir


def extract_number(s):
    return int(re.findall(r'\d+', s)[0])

# Set seed
seed = 666
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Load data
adata = sc.read_h5ad(args.input_dir + 'integrated_adata_CRCLM24.h5ad')
adata.obs['louvain'] = adata.obs['louvain'].astype('int')
adata.obs['louvain'] = adata.obs['louvain'].astype('category')

# Data preparation
sorted_batch = sorted(np.unique(adata.obs['batch_name']), key=extract_number)

# 6 statistics measures per cluster
adata        = stClinic.stClinic_Statistics_Measures(adata, sorted_batch)

# Clinical information (One-hot encoding)
All_type = []
for bid in sorted_batch:
    batch_obs = adata.obs[ adata.obs['batch_name'] == bid ]
    All_type.append( np.unique( batch_obs['type'] )[0] )
All_type = np.array(All_type)
type_idx = np.zeros([len(All_type)], dtype=int)
type_idx[All_type == 'Metastasis'] = 1
adata.uns['grading'] = type_idx

# Run stClinic for supervised prediction
adata = stClinic.train_Prediction_Model(adata, pred_type='grading', lr=args.lr_prediction, device=used_device)

# Save AnnData object
adata.write(args.out_dir + 'integrated_adata_CRCLM24-sup.h5ad', compression='gzip')  
