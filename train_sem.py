import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import sys
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
#sys.path.insert(0,'../..')
from synthetic_config import gen_args
from model_dy_syn import DynaNetGNN, HLoss
from utils import rand_int, count_parameters, Tee, AverageMeter, get_lr, to_np, set_seed
import networkx as nx
import logging
import random
from random import sample
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from scipy.special import expit as sigmoid
args = gen_args()
set_seed(args.random_seed)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
print(args)
use_gpu = torch.cuda.is_available()


# initialize model
args.stage = 'dy'
if args.stage == 'dy':

    if args.dy_model == 'gnn':
        model_dy = DynaNetGNN(args, use_gpu=use_gpu)
    else:
        raise AssertionError("Unknown dy_model %s" % args.dy_model)

    print("model_dy #params: %d" % count_parameters(model_dy))

    if args.dy_epoch >= 0:
        # if resume from a pretrained checkpoint
        model_dy_path = os.path.join(
            args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % (args.dy_epoch, args.dy_iter))
        print("Loading saved ckp for dynamics net from %s" % model_dy_path)
        model_dy.load_state_dict(torch.load(model_dy_path))

else:
    raise AssertionError("Unknown stage %s" % args.stage)
# criterion
criterionMSE = nn.MSELoss()
criterionH = HLoss()

# optimizer
if args.stage == 'dy':
    params = model_dy.parameters()
else:
    raise AssertionError('Unknown stage %s' % args.stage)
optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2, verbose=True)
if use_gpu:
    #model_kp = model_kp.cuda()
    criterionMSE = criterionMSE.cuda()

    if args.stage == 'dy':
        model_dy = model_dy.cuda()
    else:
        raise AssertionError("Unknown stage %s" % args.stage)

best_valid_loss = np.inf
save_folder = 'sythetic_interventionT5_graphfix'
save_folder = args.sv
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
writer = SummaryWriter('{}/'.format(save_folder))

# get data


def simulate_linear_sem_intervention(W, X, n, sem_type, noise_scale=None, target=None):
        """
        Simulate samples from linear SEM with specified type of noise.
        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.
        Parameters
        ----------
        W: np.ndarray
            [d, d] weighted adj matrix of DAG.
        X: np.array
            [d, 1] current X avlue.
        n: int
            Number of samples, n=inf mimics population risk.
        sem_type: str
            gauss, exp, gumbel, uniform, logistic.
        noise_scale: float
            Scale parameter of noise distribution in linear SEM.
        target: int
            Intervention target(s) for active interventions
        Return
        ------
        X: np.ndarray
            [n, d] sample matrix, [d, d] if n=inf
        """
        def _simulate_single_equation(X, w, scale):

            """X: [n, num of parents], w: [num of parents], x: [n]"""
            #print(X.shape, w.shape,'single equation')
            if sem_type == 'gauss':
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z

            elif sem_type == 'exp':
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            elif sem_type == 'logistic':
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            else:
                raise ValueError('Unknown sem type. In a linear model, \
                                 the options are as follows: gauss, exp, \
                                 gumbel, uniform, logistic.')
            #print(x)
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)

        G_nx =  nx.from_numpy_matrix(W, create_using=nx.DiGraph)
        if not nx.is_directed_acyclic_graph(G_nx):
            raise ValueError('W must be a DAG')
        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type == 'gauss':
                # make 1/d X'X = true cov
                X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
                return X
            else:
                raise ValueError('population risk not available')
        # empirical risk
       # print(G_nx)
        ordered_vertices = list(nx.topological_sort(G_nx))
        #print(ordered_vertices)
        if target == None:
            target = np.random.choice(len(ordered_vertices)-1, 1)
        else:
            target = target.cpu().numpy()
        assert len(ordered_vertices) == d
        # initialization:
        X[target] += np.random.normal(scale=1, size=1)
        rX = np.zeros([n, d])
        rX[0, :] = X


        for j in ordered_vertices:
            parents = list(G_nx.predecessors(j))
            #print(parents,'parents',j)

            rX[:, j] = _simulate_single_equation(rX[:, parents], W[parents, j], scale_vec[j])

        #print(rX.shape,'x', W.shape)
        return rX


T = 100
weighted_random_dag = DAG.erdos_renyi(n_nodes=15, n_edges=args.density*15,
                                        weight_range=(0.5, 2.0), seed=1)
dataset = IIDSimulation(W=weighted_random_dag, n=T, method='linear',
                          sem_type='gauss')
Graph, T_data = dataset.B, dataset.X

# parameter
int_t = 10
training_steps = 6000
n_kp = args.n_kp
B = 1
T_data = torch.Tensor(T_data).cuda().float()
Graph = torch.Tensor(Graph).cuda().float()
gt_graph = Graph
device = torch.device('cuda')
writer = SummaryWriter('{}/'.format(save_folder))
# train
dis_len = 70
roll_out = 10
his_len = 20
for ep in range(training_steps):
     if ep % 50 == 0:
        weighted_random_dag = DAG.erdos_renyi(n_nodes=n_kp, n_edges=1*n_kp,
                                        weight_range=(0.5, 2.0), seed=1)
        dataset = IIDSimulation(W=weighted_random_dag, n=T, method='linear',
                          sem_type='gauss')
        Graph, T_data = dataset.B, dataset.X
        T_data = torch.Tensor(T_data).cuda().float()
        Graph = torch.Tensor(Graph).cuda().float()
        gt_graph = Graph

     ''' causal discovery '''
     cur_identify = T_data[:dis_len].to(device)
     cur_graph = model_dy.graph_inference(cur_identify)

     edge_attr, edge_type_logits = cur_graph[1], cur_graph[3]
     ''' select target '''
     edge_type_logits_ = edge_type_logits.view(B,  n_kp, n_kp, args.edge_type_num)
     logits_softmax = torch.argmax(F.softmax(edge_type_logits_, dim=-1), dim=-1)
     idx = torch.arange(0, n_kp, out=torch.LongTensor())
     logits_softmax[:, idx, idx] = 10
     num_targets = 1
     values, indices = torch.topk(logits_softmax.view(B, n_kp * n_kp), k=num_targets, largest=False)
     ball_x = torch.floor_divide(indices, n_kp)
     ball_y = indices % n_kp
     ball_target = torch.cat([ball_x, ball_y], dim=-1)
     ''' roll out '''
     avg_regloss = []
     avg_TPR = []
     avg_TNR = []
     avg_shd = []
     avg_acc = []
     loss_mse = 0.
     kps_dy = T_data[dis_len:]
     for i in range(roll_out-1):

         cur_his = T_data[i + dis_len : i + dis_len + his_len].cuda()
         kp_des = kps_dy[i  + his_len].cuda() # one step

         # conduct intervention:
         if i % 20 == 0:
            cur_frame = cur_his[-1]
            #print(cur_frame.shape, i, ball_target,'target')
            #int_data, _ = generator_intervention(cur_frame, i+time_lag, ball_target[0], T=int_t)
            #print(int_data.shape, 'int', cur_his.shape)
            # combine data
            if args.active:
                # active intervention
                int_data = simulate_linear_sem_intervention(weighted_random_dag, cur_his[-1].clone().cpu().numpy(), 15, 'gauss', target=ball_target[0])
            else:
                # random intervention
                int_data = simulate_linear_sem_intervention(weighted_random_dag, cur_his[-1].clone().cpu().numpy(), 15, 'gauss', target=None)
            cur_identify = torch.cat([torch.Tensor(int_data).cuda(), cur_his], 0)
            cur_graph = model_dy.graph_inference(cur_identify)
            ''' select active intervention target '''
            edge_attr, edge_type_logits = cur_graph[1], cur_graph[3]
            edge_type_logits_ = edge_type_logits.view(B,  n_kp, n_kp, args.edge_type_num)
            logits_softmax = torch.argmax(F.softmax(edge_type_logits_, dim=-1), dim=-1)
            idx = torch.arange(0, 3, out=torch.LongTensor())
            logits_softmax[:, idx, idx] = 10
            num_targets = 1
            values, indices = torch.topk(logits_softmax.view(B, n_kp * n_kp), k=num_targets, largest=False)
            ball_x = torch.floor_divide(indices, n_kp)
            ball_y = indices % n_kp
            ball_target = torch.cat([ball_x, ball_y], dim=-1).cuda()

         # kp_pred: B x n_kp x 1

         cur_his = cur_his[None, : , :,  None]
         kp_pred = model_dy.dynam_prediction(cur_his, cur_graph)
         mean_cur = kp_pred[:, :, :1] #, kp_pred[:, :, 2:].view(B, n_kp, 2, 2)
         mean_des = kp_des[None, : , None] #, covar_gt[:, 0].view(B, n_kp, 2, 2)
         loss_mse_cur = criterionMSE(mean_cur, mean_des).cuda()
         loss_mse += loss_mse_cur / roll_out
         avg_regloss.append(loss_mse.item())
         if i % 100 == 0:
             # Accuracy
             idx_pred = torch.argmax(edge_type_logits, dim=3).data.cpu().numpy()

             acc = np.logical_and(idx_pred[0] == gt_graph.data.cpu().numpy(), np.logical_not(np.eye(n_kp)))
             acc = np.sum(acc) / (B * n_kp * (n_kp - 1))
             #print(gt_graph.shape,'gt',gt_graph)
             avg_acc.append(acc)
             confusion_vector =  torch.argmax(edge_type_logits, dim=3)/ gt_graph
             mt = MetricsDAG(idx_pred[0], gt_graph.data.cpu().numpy())
             shd = mt.metrics['shd']
             avg_shd.append(shd)
             if args.shd:
                loss_mse += shd/100
             else:
                 pass
     optimizer.zero_grad()
     loss_mse.backward()
     optimizer.step()



     if ep % 500 == 0:
          torch.save(model_dy.state_dict(), '%s/net_dy_epoch_%d.pth' % (save_folder, ep))
     if ep % 20 == 0:
       print('epoch:',ep,np.mean(avg_regloss),'mse', np.mean(avg_acc),'avg accuracy', np.mean(avg_shd),'avg SHD')
     #print('TPR:',np.mean(TPR),'TNR',np.mean(TNR),'F1',np.mean(f1))
     writer.add_scalar("Train/avg_regloss", np.mean(avg_regloss), ep)
     writer.add_scalar("Train/avg_acc", np.mean(avg_acc), ep)
     #writer.add_scalar("Train/avg_TPR", np.mean(avg_TPR), ep)
     #writer.add_scalar("Train/avg_TNR", np.mean(avg_TNR), ep)
     writer.add_scalar("Train/avg_shd", np.mean(avg_shd), ep)


