import pandas as pd
import sklearn.neighbors
import scipy.sparse as sp
import networkx as nx
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
import faiss

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri, rpy2.robjects.pandas2ri

from typing import List, Mapping, Optional, Union
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData
from sklearn.preprocessing import scale
from scipy.sparse.linalg import eigsh
from sklearn.metrics import silhouette_score as s_score
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

def parameter_setting():

    parser = argparse.ArgumentParser(description='stClinic for predicting clinically-related niches')
    parser.add_argument('--input_dir',   '-IP', type = str, default = '/home/zuocm/Share_data/xiajunjie/stClinic/Datasets/DLPFC/',  help='data directory')
    parser.add_argument('--out_dir',   '-OP', type = str, default = '/home/zuocm/Share_data/xiajunjie/stClinic/Datasets/DLPFC/stClinic/',  help='data directory')
    parser.add_argument('--rad_cutoff',   '-RC', type = int, default = 150,    help='The radius value used to construct intra-edge (i.e., spatial nearest neighbors) for each slice')
    parser.add_argument('--k_cutoff',   '-KC', type = int, default = 6,    help='The number of spatial neighbors used to construct intra-edge')
    parser.add_argument('--k',   '-K', type = int, default = 5,    help='The number of mutual nearest neighbors used to construct inter-edges across slices')
    parser.add_argument('--n_top_genes',   '-NTG', type = int, default = 5000,    help='The number of highly variable genes selected for each slice')
    parser.add_argument('--n_centroids',   '-NCS', type = int, default = 7,    help='The number of components of the GMM')
    parser.add_argument('--lr_integration',   '-LRI', type = float, default = 0.0005,    help='The learning rate used by stClinic when extracting batch-corrected features in slices')
    parser.add_argument('--lr_prediction',   '-LRP', type = float, default = 0.05,    help='The learning rate used by supervised stClinic')

    return parser


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating intra-spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per spot on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='VGAEX', random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    robjects.r.library("mclust")

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    mclust_res = np.array(rmclust(np.array(adata.obsm[used_obsm]), num_cluster, modelNames)[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def match_cluster_labels(true_labels,est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i+1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j-1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr==org_cat[i])* (est_labels_arr==est_cat[j]))
            B.add_edge(i+1,-j-1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
#     match = minimum_weight_full_matching(B)
    if len(org_cat)>=len(est_cat):
        return np.array([match[-est_cat.index(c)-1]-1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c)-1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c)-1) in match: 
                l.append(match[-est_cat.index(c)-1]-1)
            else:
                l.append(len(org_cat)+unmatched.index(c))
        return np.array(l)  



def count_params(model):
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f'Total params: {all_params}\nTrainable params: {trainable_params}\nNon-trainable params: {non_trainable_params}')


def estimate_k(data):
    """
    Estimate number of groups k:
        based on random matrix theory (RTM), borrowed from SC3
        input data is (p,n) matrix, p is feature, n is sample
    """
    p, n = data.shape

    x = scale(data, with_mean=False)
    muTW = (np.sqrt(n-1) + np.sqrt(p)) ** 2
    sigmaTW = (np.sqrt(n-1) + np.sqrt(p)) * (1/np.sqrt(n-1) + 1/np.sqrt(p)) ** (1/3)
    sigmaHatNaive = x.T.dot(x)

    bd = np.sqrt(p) * sigmaTW + muTW
    evals, _ = eigsh(sigmaHatNaive)

    k = 0
    for i in range(len(evals)):
        if evals[i] > bd:
            k += 1
    return k


def F1score(adata):
    """
    Compute silhouette coefficient:
    F1score = 2*(1-silh'batch)*silh'cluster / [silh'cluster + (1-silh'batch)]
    silh'batch = (1+silh_batch)/2
    silh'cluster = (1+silh_cluster)/2
    """
    s_cluster = s_score(adata.obsm['X_umap'], adata.obs['mclust'])
    s_batch   = s_score(adata.obsm['X_umap'], adata.obs['batch_name'])

    s_cluster = (1+s_cluster)/2
    s_batch   = (1+s_batch)/2

    F1score   = 2*(1-s_batch)*s_cluster / (s_cluster + (1-s_batch))

    return F1score



def LISI_score(adata, random_seed=666):
    """
    Compute cell-type / integration local inverse simpson'index
    """

    np.random.seed(random_seed)
    robjects.r.library("lisi")
    rlisi = robjects.r['compute_lisi']

    rpy2.robjects.numpy2ri.activate()
    rpy2.robjects.pandas2ri.activate()

    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)

    robjects.globalenv['X'] = np.array(adata.obsm['X_umap'])
    meta_data = adata.obs[['Ground Truth', 'batch_name']]
    meta_data.columns = ['label1','label2']
    robjects.globalenv['meta_data'] = meta_data
    robjects.globalenv['cnames'] = robjects.r['c']('label1', 'label2')

    lisi_res = np.array(
                        rlisi(robjects.globalenv['X'], robjects.globalenv['meta_data'], robjects.globalenv['cnames'])
                        )

    clisi = np.mean(lisi_res, axis=1)[0]
    ilisi = np.mean(lisi_res, axis=1)[1]

    return dict(cLISI = clisi, iLISI = ilisi, LISI=lisi_res)


def CIndex_lifeline(hazards, labels, survtime_all):
    return(concordance_index(survtime_all, -hazards, labels))


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def KM_plot(hazardsdata, labels, survtime_all, output_dir):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = (hazards_dichotomize==1)

    T = survtime_all
    E = labels

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    kmf_high = KaplanMeierFitter()
    ax = kmf_high.fit(T[idx], E[idx], label='high risk (n={})'.format(sum(idx))).plot_survival_function(linewidth=3, show_censors=True, censor_styles={'ms':9, 'marker':'+'}, ax=ax, ci_show=False, color='#FF4B68')

    kmf_low = KaplanMeierFitter()
    ax = kmf_low.fit(T[~idx], E[~idx], label='low risk (n={})'.format(len(idx)-sum(idx))).plot_survival_function(linewidth=3, show_censors=True, censor_styles={'ms':9, 'marker':'+'}, ax=ax, ci_show=False, color='#118DF0')

    legend = plt.legend(loc="upper right",fontsize=10)
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_linewidth(0.0)
    plt.xlabel('Time (days)',fontsize=12)
    plt.ylim([-0.01, 1.01])
    plt.ylabel('Population at risk (%)',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    logrank_stat = logrank_test(T[idx], T[~idx], E[idx], E[~idx], alpha=.99).p_value
    plt.text(7.5,0.05,s='p={:.2e}'.format(logrank_stat),fontsize=10)

    cph = CoxPHFitter()
    df = pd.DataFrame({'Surv_time': T, 'Event':T, 'High_risk':hazards_dichotomize})
    cph.fit(df, 'Surv_time', 'Event')
    hr_value = cph.summary.loc['High_risk', 'exp(coef)']
    ci_lower = cph.summary.loc['High_risk', 'exp(coef) lower 95%']
    ci_upper = cph.summary.loc['High_risk', 'exp(coef) upper 95%']

    plt.text(7.5,0.07, s=f'HR: {hr_value:.3f} ({ci_lower:.3f}, {ci_upper:.3f})', fontsize=10)

    plt.savefig(output_dir + 'Stat_AttPred_UMAP_KM_plot.jpg', dpi=500)



def CM_plot(pred_data, labels, output_dir):
    cm = confusion_matrix(labels, pred_data)

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap='OrRd', fmt='g', annot_kws={'fontsize': 12})
    plt.xlabel('Ground-truth label', fontdict={'size':12})
    plt.ylabel('Predicted label', fontdict={'size':12})

    plt.savefig(output_dir + 'Stat_AttPred_UMAP_CM_plot.jpg', dpi=500)



def ROC_plot(pred_data, labels, output_dir):
    fpr, tpr, thresholds = roc_curve(labels, pred_data)
    auc_score = auc(fpr,tpr)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC: {auc_score:.3f}')
    plt.plot([0,1],[0,1],linestyle='--',color='grey')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate',fontdict={'size':12})
    plt.ylabel('True Positive Rate',fontdict={'size':12})
    plt.legend(fontsize=12,loc='lower right')

    plt.savefig(output_dir + 'Stat_AttPred_UMAP_ROC_plot.jpg', dpi=500)



def knn_smoothing(latent, k, mat):
    dist = pairwise_distances(latent)
    row = []
    col = []
    sorted_knn = dist.argsort(axis=1)
    for idx in list(range(np.shape(dist)[0])):
        col.extend(sorted_knn[idx, : k].tolist())
        row.extend([idx] * k)

    res = np.zeros((mat.shape[0], mat.shape[1]))
    for i in range(len(col)):
        res[row[i]] += mat[col[i]]

    return res




def spatial_match(embds:List[torch.Tensor],
                  reorder:Optional[bool]=True,
                  smooth:Optional[bool]=True,
                  smooth_range:Optional[int]=20,
                  scale_coord:Optional[bool]=True,
                  adatas:Optional[List[AnnData]]=None,
                  return_euclid:Optional[bool]=False,
                  verbose:Optional[bool]=False,
                  get_null_distri:Optional[bool]=False
    )-> List[Union[np.ndarray,torch.Tensor]]:
    r"""
    Use embedding to match cells from different batches based on cosine similarity
    
    Parameters
    ----------
    embds
        list of embeddings
    reorder
        if reorder embedding by cell numbers
    smooth
        if smooth the mapping by Euclid distance
    smooth_range
        use how many candidates to do smooth
    scale_coord
        if scale the coordinate to [0,1]
    adatas
        list of adata object
    verbose
        if print log
    get_null_distri
        if get null distribution of cosine similarity
    
    Note
    ----------
    Automatically use larger dataset as source
    
    Return
    ----------
    Best matching, Top n matching and cosine similarity matrix of top n  
    """
    if reorder and embds[0].shape[0] < embds[1].shape[0]:
        embd0 = embds[1]
        embd1 = embds[0]
        adatas = adatas[::-1] if adatas is not None else None
    else:
        embd0 = embds[0]
        embd1 = embds[1]
        
    if get_null_distri:
        embd0 = torch.tensor(embd0)
        embd1 = torch.tensor(embd1)
        sample1_index = torch.randint(0, embd0.shape[0], (1000,))
        sample2_index = torch.randint(0, embd1.shape[0], (1000,))
        cos = torch.nn.CosineSimilarity(dim=1)
        null_distri = cos(embd0[sample1_index], embd1[sample2_index])

    index = faiss.index_factory(embd1.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    embd0_np = embd0.detach().cpu().numpy() if torch.is_tensor(embd0) else embd0
    embd1_np = embd1.detach().cpu().numpy() if torch.is_tensor(embd1) else embd1
    embd0_np = embd0_np.copy().astype('float32')
    embd1_np = embd1_np.copy().astype('float32')
    faiss.normalize_L2(embd0_np)
    faiss.normalize_L2(embd1_np)
    index.add(embd0_np)
    similarity, order = index.search(embd1_np, smooth_range)
    best = []
    if smooth and adatas != None:
        if verbose:
            print('Smoothing mapping, make sure object is in same direction')
        if scale_coord:
            # scale spatial coordinate of every adata to [0,1]
            adata1_coord = adatas[0].obsm['spatial'].copy()
            adata2_coord = adatas[1].obsm['spatial'].copy()
            for i in range(2):
                    adata1_coord[:,i] = (adata1_coord[:,i]-np.min(adata1_coord[:,i]))/(np.max(adata1_coord[:,i])-np.min(adata1_coord[:,i]))
                    adata2_coord[:,i] = (adata2_coord[:,i]-np.min(adata2_coord[:,i]))/(np.max(adata2_coord[:,i])-np.min(adata2_coord[:,i]))
        dis_list = []
        for query in range(embd1_np.shape[0]):
            ref_list = order[query, :smooth_range]
            dis = euclidean_distances(adata2_coord[query,:].reshape(1, -1),
                                      adata1_coord[ref_list,:])
            dis_list.append(dis)
            best.append(ref_list[np.argmin(dis)])
    else:
        best = order[:,0]

    if return_euclid and smooth and adatas != None:
        dis_array = np.squeeze(np.array(dis_list))
        if get_null_distri:
            return np.array(best), order, similarity, dis_array, null_distri
        else:
            return np.array(best), order, similarity, dis_array
    else:
        return np.array(best), order, similarity



class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, checkpoint_file=''):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 10
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        if self.checkpoint_file:
            torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss
