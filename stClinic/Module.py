import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import math
import re
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim as optim

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
from tqdm.autonotebook import trange
from torch_geometric.loader import DenseDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .Utilities import *
from .gat_conv import GATConv

cudnn.deterministic = True
cudnn.benchmark = True


class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization for each slice
    """
    def __init__(self, num_features, n_domains, eps=1e-5, momentum=0.1):
        super().__init__()
        self.n_domains = n_domains
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domains)])
        
    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()
            
    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()
            
    def _check_input_dim(self, input):
        raise NotImplementedError
            
    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)
        for i in range(self.n_domains):
            indices = np.where(y.cpu().numpy()==i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]

        return out



class stClinic(nn.Module):
    def __init__(self, hidden_dims_integ, n_domains, n_centroids):
        super(stClinic, self).__init__()

        # Set network structure
        [in_dim, num_hidden, out_dim] = hidden_dims_integ

        self.fc1      = nn.Linear(in_dim, in_dim)
        self.norm     = nn.BatchNorm1d(in_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.conv1 = GATConv(in_dim, num_hidden, heads=3, concat=False,
                             dropout=0.2, add_self_loops=False, bias=False)

        self.conv_mu = GATConv(num_hidden, out_dim, heads=3, concat=False,
                               dropout=0.2, add_self_loops=False, bias=False)
        self.conv_log_var = GATConv(num_hidden, out_dim, heads=3, concat=False,
                                    dropout=0.2, add_self_loops=False, bias=True)

        self.fc2      = nn.Linear(out_dim, in_dim)
        self.dsbnorm  = DSBatchNorm(in_dim, n_domains)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(in_dim, in_dim)

        self.graph_decoder = InnerProductDecoder()

        # init c parameters
        self.n_centroids = n_centroids

        self.pi    = nn.Parameter(torch.ones(n_centroids)/n_centroids) # p(c)
        self.mu_c  = nn.Parameter(torch.zeros(out_dim, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(out_dim, n_centroids)) # sigma^2

    def reparameterize(self, mu, log_var):
        if self.training:
            return mu + torch.randn_like(log_var) * torch.exp(log_var.mul(0.5))
        else:
            return mu

    def get_gamma(self, z):
        """
        Inference c from z

        gamma equals q(c|x)

        q(c|x) = p(c|z) = p(c)*p(z|c)/p(z)
        """
        n_centroids = self.n_centroids

        N     = z.size(0)
        z     = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi    = self.pi.repeat(N, 1) # N*K
        mu_c  = self.mu_c.repeat(N, 1, 1) # N*D*K
        var_c = self.var_c.repeat(N, 1, 1) + 1e-8 # N*D*K

        # p(c, z) = p(c)*p(z|c)
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5 * torch.log(2 * math.pi * var_c) + (z - mu_c)**2/(2 * var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, features, edge_index, adj_mat, type='GMM', seed=666, start_stage=False):
        """
        Initialize stClinic integration model with GMM model parameters / Prune edges
        """
        if type == 'GMM':

            gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag', random_state=seed)

            h0 = self.dropout1( F.elu( self.norm( self.fc1(features) ) ) )

            h1 = F.elu(self.conv1(h0, edge_index, attention=True))

            mu = self.conv_mu(h1, edge_index, attention=True)
            log_var = self.conv_log_var(h1, edge_index, attention=True)

            z = mu
            z = z.cpu().detach().numpy()

            gmm.fit(z)
            labels = gmm.predict(z)

            if start_stage:
                print('Selected initial type: GMM')

                '''
                Variance Reduction
                '''
                sampled_z = []
                for c in range(self.n_centroids):
                    z_c = z[labels==c,:]
                    mean_c = gmm.means_[c,:]
                    sampled_z_c = z_c[np.argsort(np.sum((z_c - mean_c)**2, axis=1)) <= int(z_c.shape[0]/2),:]
                    sampled_z.append(sampled_z_c)
                sampled_z = np.concatenate(sampled_z, axis=0)

                sampled_gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag', random_state=seed)
                sampled_gmm.fit(sampled_z)

                self.mu_c.data.copy_(torch.from_numpy(sampled_gmm.means_.T.astype(np.float32)))
                self.var_c.data.copy_(torch.from_numpy(sampled_gmm.covariances_.T.astype(np.float32)))

            else:
                print('Prune edges...')

                prune_mat = torch.IntTensor(np.where(pairwise_distances(labels.reshape(-1,1), metric='hamming')==0, 1, 0)).cuda()
                pruned_adj = prune_mat.to(torch.int8) * adj_mat.to(torch.int8)

                return pruned_adj

        elif type == 'KMeans':

            h0 = self.dropout1( F.elu( self.norm( self.fc1(features) ) ) )

            h1 = F.elu(self.conv1(h0, edge_index, attention=True))

            mu = self.conv_mu(h1, edge_index, attention=True)
            log_var = self.conv_log_var(h1, edge_index, attention=True)

            z = mu

            kmeans = KMeans(n_clusters=self.n_centroids, random_state=seed).fit(z.cpu().detach().numpy())
            mean = kmeans.cluster_centers_
            cls_index = kmeans.labels_

            if start_stage:
                print('Selected initial type: KMeans')

                '''
                Variance Reduction
                '''
                sampled_z = []
                for c in range(self.n_centroids):
                    z_c = z.cpu().detach().numpy()[cls_index==c,:]
                    mean_c = mean[c,:]
                    sampled_z_c = z_c[np.argsort(np.sum((z_c - mean_c)**2, axis=1)) <= int(z_c.shape[0]/2),:]
                    sampled_z.append(sampled_z_c)
                sampled_z = np.concatenate(sampled_z, axis=0)

                sampled_kmeans = KMeans(n_clusters=self.n_centroids, random_state=seed).fit(sampled_z)
                sampled_mean = sampled_kmeans.cluster_centers_
                sampled_cls_index = sampled_kmeans.labels_

                sampled_var = []
                sampled_mean = torch.from_numpy(sampled_mean).cuda()
                sampled_z = torch.from_numpy(sampled_z).cuda()
                for c in range(self.n_centroids):
                    index = np.where(sampled_cls_index==c)
                    var_g = torch.sum((sampled_z[index[0],:]-sampled_mean[c,:])**2, dim=0, keepdim=True)/(len(index[0])-1)
                    sampled_var.append(var_g)
                sampled_var = torch.cat(sampled_var, dim=0)
                sampled_mean = sampled_mean.cpu().detach().numpy().astype(np.float32)
                sampled_var = sampled_var.cpu().detach().numpy().astype(np.float32)

                self.mu_c.data.copy_(torch.from_numpy(sampled_mean.transpose()))
                self.var_c.data.copy_(torch.from_numpy(sampled_var.transpose()))
            
            else:
                print('Prune edges...')

                prune_mat = torch.IntTensor(np.where(pairwise_distances(cls_index.reshape(-1,1), metric='hamming')==0, 1, 0)).cuda()
                pruned_adj = prune_mat.to(torch.int8) * adj_mat.to(torch.int8)

                return pruned_adj

    def forward(self, features, edge_index, y):

        h0 = self.dropout1( F.elu( self.norm( self.fc1(features) ) ) )

        h1 = F.elu(self.conv1(h0, edge_index, attention=True))

        mu      = self.conv_mu(h1, edge_index, attention=True)
        log_var = self.conv_log_var(h1, edge_index, attention=True)

        z  = self.reparameterize(mu, log_var)

        h2 = self.dropout2( F.elu( self.dsbnorm( self.fc2(z), y ) ) )

        h3 = self.fc3(h2)

        return z, mu, log_var, h3
    
    def graph_recon_loss(self, z, pos_edge_label_index):

        EPS = 1e-15

        neg_edge_index = None

        reG = self.graph_decoder(z, pos_edge_label_index, sigmoid=True)
        pos_loss = -torch.log(reG + EPS).mean()
        pos_edge_index, _ = remove_self_loops(pos_edge_label_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.graph_decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        graph_recon_loss = pos_loss + neg_loss

        return graph_recon_loss
    
    def KL_loss(self, z, mu, log_var):

        gamma, mu_c, var_c, pi = self.get_gamma(z)
        var_c += 1e-8
        n_c = pi.size(1)
        mu_expand = mu.unsqueeze(2).expand(mu.size(0), mu.size(1), n_c)
        logvar_expand = log_var.unsqueeze(2).expand(log_var.size(0), log_var.size(1), n_c)
        
        # log p(z|c)
        logpzc = -0.5*torch.sum(gamma*torch.sum(math.log(2*math.pi) + \
                                                torch.log(var_c) + \
                                                torch.exp(logvar_expand)/var_c + \
                                                (mu_expand-mu_c)**2/var_c, dim=1), dim=1)

        # log p(c)
        logpc = torch.sum(gamma*torch.log(pi), 1)

        # log q(z|x)
        qentropy = -0.5*torch.sum(1+log_var, 1)

        # log q(c|x)
        logqcx = torch.sum(gamma*torch.log(gamma), 1)

        # torch.mean or (1 / data.x.shape[0]) * torch.sum
        KL_loss = torch.mean( -logpzc - logpc + qentropy + logqcx )

        return KL_loss

    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)



def train_stClinic_model(adata, lr=0.0005, n_centroids=7, MultiOmics_mode=False,
                         used_feat='X_seurat', hidden_dims_integ=[512, 10],
                         n_epochs_pre=300, n_epochs_tune=300+100, n_epochs_prune=50,
                         batch_name='batch_name', lambda_KL_stage1=0.5, lambda_KL_stage2=0.2,
                         gradient_clipping=5, weight_decay=0.00001, key_add='stClinic',
                         random_seed=666, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adj_mat  = torch.IntTensor(adata.uns['adj']).to(device)
    edgeList = adata.uns['edgeList']

    if MultiOmics_mode:

        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.obsm[used_feat]),
                    y=torch.LongTensor(adata.obs[batch_name].cat.codes))

    else:

        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    prune_edge_index=torch.LongTensor(np.array([])),
                    x=torch.FloatTensor(adata.X.todense()),
                    y=torch.LongTensor(adata.obs[batch_name].cat.codes))

    # Transform
    transform  = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, add_negative_train_samples=False, split_labels=True)
    data, _, _ = transform(data)
    data       = data.to(device)

    model = stClinic(hidden_dims_integ=[data.x.shape[1], hidden_dims_integ[0], hidden_dims_integ[1]],
                     n_domains=len(adata.obs[batch_name].cat.categories),
                     n_centroids=n_centroids).to(device)

    print('Pretrain with unregularized stClinic (without GMM regularizer)...')
    model.pi.requires_grad, model.mu_c.requires_grad, model.var_c.requires_grad = False, False, False

    optimizer_stClinic = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    with trange(n_epochs_pre, total=n_epochs_pre, desc='Epochs') as tq:
        for epoch in tq:
            model.train()

            optimizer_stClinic.zero_grad()

            z, mu, log_var, reX = model(data.x, data.edge_index, data.y)

            features_recon_loss = F.mse_loss(data.x, reX)

            graph_recon_loss = model.graph_recon_loss(mu, data.pos_edge_label_index)

            KL_loss = 0

            epoch_loss = {'feat_recon_loss':features_recon_loss*10, 'graph_recon_loss':graph_recon_loss, 'KL_loss':KL_loss*0.0}

            sum(epoch_loss.values()).backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), gradient_clipping)
            optimizer_stClinic.step()

            epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
            tq.set_postfix_str(epoch_info) 

    model.eval()

    print('Initialize Gaussian Mixture Model...')
    model.init_gmm_params(data.x, data.edge_index, adj_mat, type='GMM', seed=random_seed, start_stage=True)

    print('Train with regularized stClinic (with GMM regularizer)...')
    model.pi.requires_grad, model.mu_c.requires_grad, model.var_c.requires_grad = True, True, True

    with trange(n_epochs_tune, total=n_epochs_tune, desc='Epochs') as tq:
        for epoch in tq:

            if epoch % n_epochs_prune == 0 and epoch >= 300:
                model.eval()
                pruned_adj = model.init_gmm_params(data.x, data.edge_index, adj_mat, type='GMM', seed=random_seed, start_stage=False)
                pruned_edgeList = np.nonzero(pruned_adj.cpu().detach().numpy())

                if MultiOmics_mode:

                    data = Data(edge_index=torch.LongTensor(np.array([pruned_edgeList[0], pruned_edgeList[1]])),
                                prune_edge_index=torch.LongTensor(np.array([])),
                                x=torch.FloatTensor(adata.obsm[used_feat]),
                                y=torch.LongTensor(adata.obs[batch_name].cat.codes))

                else:

                    data = Data(edge_index=torch.LongTensor(np.array([pruned_edgeList[0], pruned_edgeList[1]])),
                                prune_edge_index=torch.LongTensor(np.array([])),
                                x=torch.FloatTensor(adata.X.todense()),
                                y=torch.LongTensor(adata.obs[batch_name].cat.codes))

                # Transform
                transform  = T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True, add_negative_train_samples=False, split_labels=True)
                data, _, _ = transform(data)

                data = data.to(device)

            model.train()

            optimizer_stClinic.zero_grad()

            z, mu, log_var, reX = model(data.x, data.edge_index, data.y)

            # Compute features reconstruction loss  --default coefficient=10 
            features_recon_loss = F.mse_loss(data.x, reX)
            # print(f'MSE loss: {features_recon_loss}')

            # Compute graph reconstruction loss
            graph_recon_loss    = model.graph_recon_loss(mu, data.pos_edge_label_index)
            # print(f'BCE loss: {graph_recon_loss}')

            # Compute KL loss  --coefficient=0.0(like STAGATE)  --coefficient=0.5 (stage I)  --coefficient=0.2 (stage II)
            KL_loss = model.KL_loss(z, mu, log_var)
            # print(f'KL loss: {KL_loss}')

            if epoch < 300:
                epoch_loss = {'feat_recon_loss':features_recon_loss*10, 'graph_recon_loss':graph_recon_loss, 'KL_loss':KL_loss*lambda_KL_stage1}
            else:
                epoch_loss = {'feat_recon_loss':features_recon_loss*10, 'graph_recon_loss':graph_recon_loss, 'KL_loss':KL_loss*lambda_KL_stage2}

            sum(epoch_loss.values()).backward()
            nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), gradient_clipping)
            optimizer_stClinic.step()

            epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
            tq.set_postfix_str(epoch_info) 

            # early_stopping(sum(epoch_loss.values()).cpu().detach().numpy(), model)
            # if early_stopping.early_stop:
            #     print('EarlyStopping: run {} epoch'.format(epoch+1))
            #     break

    model.eval()

    z, mu, _, _ = model(data.x, data.edge_index, data.y)
    adata.obsm[key_add] = mu.cpu().detach().numpy()
    gamma, _, _, _ = model.get_gamma(z)
    adata.obs['GMM_cluster'] = np.argmax(gamma.cpu().detach().numpy(), axis=1)
    adata.obs['GMM_cluster'] = adata.obs['GMM_cluster'].astype('category')

    return adata



def stClinic_Statistics_Measures(adata, sorted_batch):

    # Statistical representations of all slices
    All_cluster_repr = []
    for bid in sorted_batch:

        batch_obs = adata.obs[ adata.obs['batch_name'] == bid ]
        batch_repr = adata.obsm['X_umap'][ adata.obs['batch_name'] == bid, :]

        res_obs = adata.obs[ adata.obs['batch_name'] != bid ]

        # Statistical representations of each slice (Before transpose)
        batch_cluster_repr = []
        for cluster in np.unique(adata.obs['louvain']):
            if cluster in np.unique(batch_obs['louvain']):

                repr_init = batch_repr[ batch_obs['louvain'] == cluster, :]

                repr_mean = np.mean( repr_init, axis=0 )
                repr_var = np.mean(repr_init * repr_init, axis=0) - repr_mean * repr_mean

                repr_L2 = np.sum(repr_init * repr_init,axis=1)
                repr_max, repr_min = repr_init[np.argmax(repr_L2),:], repr_init[np.argmin(repr_L2),:]

                repr_p1 = np.array([repr_init.shape[0] / batch_repr.shape[0]])

                repr = np.concatenate([repr_mean, repr_var, repr_max, repr_min, repr_p1])

            else:
                repr = np.zeros(9)

            repr_p2 = np.array([res_obs[res_obs['louvain']==cluster].shape[0] / res_obs.shape[0]])

            repr = np.concatenate([repr, repr_p2])

            batch_cluster_repr.append(repr)

        All_cluster_repr.append(batch_cluster_repr)

    adata.uns['Cluster_repr'] = np.array(All_cluster_repr)

    # Normalization
    adata.uns['Cluster_repr'] = np.concatenate(adata.uns['Cluster_repr'])
    adata.uns['Cluster_repr'] = (adata.uns['Cluster_repr'] - adata.uns['Cluster_repr'].mean(axis=0)) / adata.uns['Cluster_repr'].std(axis=0)
    adata.uns['Cluster_repr'] = np.reshape(adata.uns['Cluster_repr'], (len(sorted_batch), len(np.unique(adata.obs['louvain'])), adata.uns['Cluster_repr'].shape[1]))

    return adata



class stClinic_Prediction_Model(nn.Module):
    def __init__(self, hidden_dims_pred, stat_dim):
        super(stClinic_Prediction_Model, self).__init__()

        # Set attention-based prediction network structure
        [in_dim, out_dim] = hidden_dims_pred

        self.in_dim = in_dim

        self.a_list = nn.ParameterList( [nn.Parameter(torch.empty(size=(stat_dim*in_dim, 1))) for i in range(stat_dim)] )
        [ nn.init.xavier_uniform_(a.data, gain=1.414) for a in self.a_list ]

        self.leakyrelu = nn.LeakyReLU(0.2)

        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, data, pred_type):

        repr = data.x.transpose(1, 2).contiguous()

        w_list = [ torch.matmul(repr.view(repr.shape[0], -1), a.cuda()) for a in self.a_list ]
        e = self.leakyrelu( torch.cat(w_list, 1) )
        lambda_e = F.softmax(e)

        h1 = (lambda_e.unsqueeze(1).repeat(1, self.in_dim, 1).permute(0, 2, 1) * repr).sum(dim=1)

        h1 = self.softmax(h1)

        h2 = self.fc(h1)

        if pred_type=='survival':
            h2 = F.softplus(h2)
        elif pred_type=='grading':
            h2 = F.sigmoid(h2)

        return h2

    def COX_loss(self, surv_time, censor, hazard_pred):

        current_batch_len = len(surv_time)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)

        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = (surv_time[j] >= surv_time[i])

        R_mat = torch.FloatTensor(R_mat).cuda()
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)

        loss_cox = (-1) * torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)

        return loss_cox

    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)



def train_Prediction_Model(adata, hidden_dims_pred=[1],
                           pred_type='survival',
                           n_epochs=100, lr=0.014, batch_name='batch_name',
                           gradient_clipping=5, weight_decay=0.00001, key_add='Cluster_importance',
                           random_seed=666, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                           output_dir='./'):

    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    def extract_number(s):
        return int(re.findall(r'\d+', s)[0])
    sorted_batch = sorted( np.unique(adata.obs[batch_name]), key=extract_number)

    data_list = []
    for idx, bid in enumerate(sorted_batch):

        if pred_type=='survival':
            data = Data(x=torch.FloatTensor(adata.uns['Cluster_repr'][idx]),
                        y=torch.FloatTensor(adata.uns[pred_type][idx]),
                        graph_idx=bid)
        elif pred_type=='grading':
            data = Data(x=torch.FloatTensor(adata.uns['Cluster_repr'][idx]),
                        y=torch.FloatTensor([adata.uns[pred_type][idx]]),
                        graph_idx=bid)

        data = data.to(device)

        data_list.append(data)

    loader = DenseDataLoader(data_list, batch_size=8, shuffle=True)

    os.makedirs('./checkpoint/', exist_ok=True)
    early_stopping = EarlyStopping(patience=10, checkpoint_file='./checkpoint/StatAttention_UMAP_model.pt')

    model = stClinic_Prediction_Model(hidden_dims_pred=[data.x.shape[0], hidden_dims_pred[0]],
                                      stat_dim=data.x.shape[1]
                                     ).to(device)

    optimizer_pred = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    with trange(n_epochs, total=n_epochs, desc='Epochs') as tq:
        for epoch in tq:

            model.train()
            loss_total = 0

            for data in loader:

                optimizer_pred.zero_grad()

                if pred_type=='survival':
                    out = model(data, pred_type)
                    loss = model.COX_loss(data.y[:,1], data.y[:,0], out)
                elif pred_type=='grading':
                    out = model(data, pred_type)
                    loss = F.binary_cross_entropy(out, data.y)

                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), gradient_clipping)
                optimizer_pred.step()

                loss_total += loss

            epoch_info = 'loss={:.5f}'.format(loss_total)
            tq.set_postfix_str(epoch_info)

            early_stopping(loss_total.cpu().detach().numpy(), model)
            if early_stopping.early_stop:
                print('EarlyStopping: run {} epoch'.format(epoch+1))
                break

    model.eval()
    model.zero_grad()

    adata.uns[key_add] = model.fc.weight.data.detach().cpu().numpy()[0]

    test_loader = DenseDataLoader(data_list, batch_size=1, shuffle=False)
    out_total = np.array([])

    for data in test_loader:
        if pred_type=='survival':
            out = model(data, pred_type)
            out_total = np.concatenate((out_total, out.detach().cpu().numpy().reshape(-1)))
        elif pred_type=='grading':
            out = model(data, pred_type)
            pred_grading = np.array([1]) if out>=0.5 else np.array([0])
            out_total = np.concatenate((out_total, pred_grading))

    if pred_type=='survival':
        print('# ============ Prognosis prediction ============= #')

        cindex_test = CIndex_lifeline(out_total, adata.uns[pred_type][:,0], adata.uns[pred_type][:,1])
        pvalue_test = cox_log_rank(out_total, adata.uns[pred_type][:,0], adata.uns[pred_type][:,1])
        surv_acc_test = accuracy_cox(out_total, adata.uns[pred_type][:,0])
        print('CIndex: {:.3f}, Pval: {:.3f}, Acc_COX: {:.3f}'.format(cindex_test, pvalue_test, surv_acc_test))

        KM_plot(out_total, adata.uns[pred_type][:,0], adata.uns[pred_type][:,1], output_dir=output_dir)

    elif pred_type=='grading':
        print('# ============ Grading prediction ============= #')

        grading_acc = np.mean(out_total == adata.uns['grading'])
        print('Grading Acc: {:.3f}'.format(grading_acc))

        CM_plot(out_total, adata.uns['grading'], output_dir=output_dir)
        ROC_plot(out_total, adata.uns['grading'], output_dir=output_dir)

    return adata



def train_Cross_Validation_Model(adata, hidden_dims_pred=[1],
                                 pred_type='grading',
                                 n_epochs=100, lr=0.0647, batch_name='batch_name',
                                 gradient_clipping=5, weight_decay=0.00001,
                                 random_seed=666, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                                 output_dir='./'):

    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    def extract_number(s):
        return int(re.findall(r'\d+', s)[0])
    sorted_batch = sorted( np.unique(adata.obs[batch_name]), key=extract_number)

    data_list = []
    for idx, bid in enumerate(sorted_batch):

        if pred_type=='survival':
            data = Data(x=torch.FloatTensor(adata.uns['Cluster_repr'][idx]),
                        y=torch.FloatTensor(adata.uns[pred_type][idx]),
                        graph_idx=bid)
        elif pred_type=='grading':
            data = Data(x=torch.FloatTensor(adata.uns['Cluster_repr'][idx]),
                        y=torch.FloatTensor([adata.uns[pred_type][idx]]),
                        graph_idx=bid)

        data = data.to(device)

        data_list.append(data)

    if pred_type=='survival':

        # 7-fold cross-validation for prognosis prediction task

        c_index_set, acc_set = [], []

        for num_fold in range(7):

            train_idx = np.ones([len(data_list)], dtype=int)
            train_idx[ np.random.choice( np.arange(len(data_list)), int(len(data_list) / 7)+1, replace=False) ] = 0
            train_idx = train_idx==1

            train_data_list = [td for td, idx in zip(data_list, train_idx) if idx]
            test_data_list = [td for td, idx in zip(data_list, train_idx) if ~idx]

            print(f'No. {num_fold+1} fold. Train data size: {len(train_data_list)}. Test data size: {len(test_data_list)}.')

            train_loader = DenseDataLoader(train_data_list, batch_size=8, shuffle=True)

            os.makedirs('./checkpoint_CV{}/'.format((num_fold+1)), exist_ok=True)
            early_stopping = EarlyStopping(patience=10, checkpoint_file='./checkpoint_CV{}/Stat_AttPred_UMAP_model_CV{}.pt'.format((num_fold+1), (num_fold+1)))

            model = stClinic_Prediction_Model(hidden_dims_pred=[data.x.shape[0], hidden_dims_pred[0]],
                                                stat_dim=data.x.shape[1]
                                                ).to(device)

            optimizer_pred = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

            with trange(n_epochs, total=n_epochs, desc='Epochs') as tq:
                for epoch in tq:

                    model.train()
                    loss_train = 0

                    for train_data in train_loader:

                        optimizer_pred.zero_grad()

                        out = model(train_data, pred_type)
                        loss = model.COX_loss(train_data.y[:,1], train_data.y[:,0], out)

                        loss.backward()
                        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), gradient_clipping)
                        optimizer_pred.step()

                        loss_train += loss

                    epoch_info = 'train loss={:.5f}'.format(loss_train)
                    tq.set_postfix_str(epoch_info)

                    early_stopping(loss_train.cpu().detach().numpy(), model)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} epoch'.format(epoch+1))
                        break

            model.eval()
            model.zero_grad()

            test_loader = DenseDataLoader(test_data_list, batch_size=1, shuffle=False)

            out_total = np.array([])
            for test_data in test_loader:
                out = model(test_data, pred_type)
                out_total = np.concatenate((out_total, out.detach().cpu().numpy().reshape(-1)))
            
            cindex_test = CIndex_lifeline(out_total, adata.uns[pred_type][:,0][~train_idx], adata.uns[pred_type][:,1][~train_idx])
            surv_acc_test = accuracy_cox(out_total, adata.uns[pred_type][:,0][~train_idx])

            print('No. {:d} fold. COX: {:.3f}, ACC: {:.3f}'.format(num_fold+1, cindex_test, surv_acc_test))

            c_index_set.append(cindex_test)
            acc_set.append(surv_acc_test)

        c_index_avg = np.mean(c_index_set)
        acc_avg = np.mean(acc_set)

        print('7-fold cross-validation. COX: {:.3f}, ACC: {:.3f}'.format(c_index_avg, acc_avg))

        return c_index_avg, acc_avg

    elif pred_type=='grading':

        # Leava-one-out cross-validation for grading prediction task

        out_total = np.array([])

        for num_fold in range(len(data_list)):

            train_idx = np.ones([len(data_list)], dtype=int)
            train_idx[num_fold] = 0
            train_idx = train_idx==1

            train_data_list = [td for td, idx in zip(data_list, train_idx) if idx]
            test_data_list = [td for td, idx in zip(data_list, train_idx) if ~idx]

            print(f'No. {num_fold+1} fold. Train data size: {len(train_data_list)}. Test data size: {len(test_data_list)}.')

            train_loader = DenseDataLoader(train_data_list, batch_size=8, shuffle=True)

            os.makedirs('./checkpoint_CV{}/'.format((num_fold+1)), exist_ok=True)
            early_stopping = EarlyStopping(patience=10, checkpoint_file='./checkpoint_CV{}/Stat_AttPred_UMAP_model_CV{}.pt'.format((num_fold+1), (num_fold+1)))

            model = stClinic_Prediction_Model(hidden_dims_pred=[data.x.shape[0], hidden_dims_pred[0]],
                                              stat_dim=data.x.shape[1]
                                             ).to(device)

            optimizer_pred = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

            with trange(n_epochs, total=n_epochs, desc='Epochs') as tq:
                for epoch in tq:

                    model.train()
                    loss_train = 0

                    for train_data in train_loader:

                        optimizer_pred.zero_grad()

                        out = model(train_data, pred_type)
                        loss = F.binary_cross_entropy(out, train_data.y)

                        loss.backward()
                        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), gradient_clipping)
                        optimizer_pred.step()

                        loss_train += loss

                    epoch_info = 'train loss={:.5f}'.format(loss_train)
                    tq.set_postfix_str(epoch_info)

                    early_stopping(loss_train.cpu().detach().numpy(), model)
                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} epoch'.format(epoch+1))
                        break

            model.eval()
            model.zero_grad()

            test_loader = DenseDataLoader(test_data_list, batch_size=1, shuffle=False)

            for test_data in test_loader:
                out = model(test_data, pred_type)
                pred_grading = np.array([1]) if out>=0.5 else np.array([0])

            out_total = np.concatenate((out_total, pred_grading))

        grading_acc = np.mean(out_total == adata.uns['grading'])

        print('Leave-one-out cross-validation. Grading Acc: {:.3f}'.format(grading_acc))

        ROC_plot(out_total, adata.uns['grading'], output_dir=output_dir)
        CM_plot(out_total, adata.uns['grading'], output_dir=output_dir)

        return grading_acc
