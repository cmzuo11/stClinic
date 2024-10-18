import numpy as np
import pandas as pd
import itertools
import networkx as nx
import hnswlib
import scipy.linalg

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix

def create_dictionary_mnn(adata, use_rep, batch_name, k = 5, save_on_disk = True, approx = True, iter_comb = None):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm[use_rep])
        cells.append(cell_names[batch_list == i])

    batch_name_df = pd.DataFrame(np.array(batch_list.unique()))
    mnns = dict()

    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(cells)), 2))
    for comb in iter_comb:
        i = comb[0]
        j = comb[1]
        key_name1 = batch_name_df.loc[comb[0]].values[0] + "_" + batch_name_df.loc[comb[1]].values[0]
        mnns[key_name1] = {}

        new = list(cells[j])
        ref = list(cells[i])

        ds1 = adata[new].obsm[use_rep]
        ds2 = adata[ref].obsm[use_rep]

        match = mnn(ds1, ds2, knn=k, save_on_disk = save_on_disk, approx = approx)
        
        coordinates = list(match)

        sparse_matrix = coo_matrix((np.ones(len(coordinates)), (zip(*coordinates))), shape=(len(ref), len(new)))

        mnns[key_name1] = sparse_matrix
    return(mnns)

def nn_approx(ds1, ds2, knn=5):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))
    return match

def nn(ds1, ds2, knn=5, metric_p=2):

    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

def mnn(ds1, ds2, knn = 5, save_on_disk = True, approx = True):
    if approx: 
        # Find nearest neighbors in first direction.
        # output KNN point for each point in ds1.  match1 is a set(): (points in names1, points in names2), the size of the set is ds1.shape[0]*knn
        match1 = nn_approx(ds1, ds2, knn=knn)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, knn=knn)
    else:
        match1 = nn(ds1, ds2, knn=knn)
        match2 = nn(ds2, ds1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match2 & set([ (b, a) for a, b in match1 ])

    return mutual

def inter_linked_graph(adj_list, section_ids, mnn_dict):

    if mnn_dict:
        
        # inter_linked_graph

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):

            last_batch_len = adj_concat.shape[0]

            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

            aux_mat = []
            for aux_id in range(batch_id):

                key_name1 = section_ids[aux_id] + '_' + section_ids[batch_id]

                aux_mat.append(mnn_dict[key_name1].toarray())
            aux_mat = np.concatenate(aux_mat, axis=0)

            adj_concat[:last_batch_len, last_batch_len:] = aux_mat
            adj_concat[last_batch_len:, :last_batch_len] = aux_mat.T

    else:
        
        # graph without inter-slice edges

        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(section_ids)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

    return adj_concat

def count_spatial_neighbors(Batch_list, section_ids, adj_concat):

    start_num = 0
    for batch_id in range(len(Batch_list)):

        spot_num = Batch_list[batch_id].n_obs

        print(f'\n------Counting {section_ids[batch_id]} spatial neighbors...')

        avg_neigh_num = np.mean(np.sum(adj_concat[start_num:start_num+spot_num,:]==1,axis=1)-1)
        print(f'{round(avg_neigh_num,4)} neighbors per spot on average.')

        avg_intra_neigh_num = np.mean(np.sum(adj_concat[start_num:start_num+spot_num,start_num:start_num+spot_num]==1,axis=1)-1)
        print(f'{round(avg_intra_neigh_num,4)} intra-neighbors per spot on average.')

        avg_inter_neigh_num = avg_neigh_num - avg_intra_neigh_num
        print(f'{round(avg_inter_neigh_num,4)} inter-neighbors per spot on average.')

        start_num += spot_num
