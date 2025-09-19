import time

import dgl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch.nn import init

from .dgl_layers import DiffPoolBatchedGraphLayer, GraphSage, GraphSageLayer
from .model_utils import batch2tensor
from .tensorized_layers import *


class DiffPool(nn.Module):
    """
    DiffPool Fuse
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        embedding_dim,
        label_dim,
        activation,
        n_layers,
        dropout,
        n_pooling,
        linkpred,
        batch_size,
        aggregator_type,
        assign_dim,
        pool_ratio,
        custom_loss_weight,
        cat=False,
    ):
        super(DiffPool, self).__init__()
        self.link_pred = linkpred
        self.concat = cat
        self.n_pooling = n_pooling
        self.batch_size = batch_size
        self.link_pred_loss = []
        self.entropy_loss = []
        self.custom_loss_weight = custom_loss_weight

        # list of GNN modules before the first diffpool operation
        self.gc_before_pool = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()

        # list of list of GNN modules, each list after one diffpool operation
        self.gc_after_pool = nn.ModuleList()
        self.assign_dim = assign_dim
        self.bn = True
        self.num_aggs = 1

        # constructing layers
        # layers before diffpool
        assert n_layers >= 3, "n_layers too few"
        self.gc_before_pool.append(
            GraphSageLayer(
                input_dim,
                hidden_dim,
                activation,
                dropout,
                aggregator_type,
                self.bn,
            )
        )
        for _ in range(n_layers - 2):
            self.gc_before_pool.append(
                GraphSageLayer(
                    hidden_dim,
                    hidden_dim,
                    activation,
                    dropout,
                    aggregator_type,
                    self.bn,
                )
            )
        self.gc_before_pool.append(
            GraphSageLayer(
                hidden_dim, embedding_dim, None, dropout, aggregator_type
            )
        )

        assign_dims = []
        assign_dims.append(self.assign_dim)
        if self.concat:
            # diffpool layer receive pool_emedding_dim node feature tensor
            # and return pool_embedding_dim node embedding
            pool_embedding_dim = hidden_dim * (n_layers - 1) + embedding_dim
        else:
            pool_embedding_dim = embedding_dim

        self.first_diffpool_layer = DiffPoolBatchedGraphLayer(
            pool_embedding_dim,
            self.assign_dim,
            hidden_dim,
            activation,
            dropout,
            aggregator_type,
            self.link_pred,
        )
        gc_after_per_pool = nn.ModuleList()

        for _ in range(n_layers - 1):
            gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, hidden_dim))
        gc_after_per_pool.append(BatchedGraphSAGE(hidden_dim, embedding_dim))
        self.gc_after_pool.append(gc_after_per_pool)

        self.assign_dim = int(self.assign_dim * pool_ratio)
        # each pooling module
        for _ in range(n_pooling - 1):
            self.diffpool_layers.append(
                BatchedDiffPool(
                    pool_embedding_dim,
                    self.assign_dim,
                    hidden_dim,
                    self.link_pred,
                )
            )
            gc_after_per_pool = nn.ModuleList()
            for _ in range(n_layers - 1):
                gc_after_per_pool.append(
                    BatchedGraphSAGE(hidden_dim, hidden_dim)
                )
            gc_after_per_pool.append(
                BatchedGraphSAGE(hidden_dim, embedding_dim)
            )
            self.gc_after_pool.append(gc_after_per_pool)
            assign_dims.append(self.assign_dim)
            self.assign_dim = int(self.assign_dim * pool_ratio)

        # predicting layer
        if self.concat:
            self.pred_input_dim = (
                pool_embedding_dim * self.num_aggs * (n_pooling + 1)
            )
        else:
            self.pred_input_dim = embedding_dim * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, label_dim)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def gcn_forward(self, g, h, gc_layers, cat=False):
        """
        Return gc_layer embedding cat.
        """
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h = gc_layer(g, h)
            block_readout.append(h)
        h = gc_layers[-1](g, h)
        block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=1)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def forward(self, g):
        self.link_pred_loss = []
        self.entropy_loss = []
        h = g.ndata["feat"] # [21352, 7]
        # node feature for assignment matrix computation is the same as the
        # original node feature
        h_a = h

        out_all = []

        # we use GCN blocks to get an embedding first
        g_embedding = self.gcn_forward(g, h, self.gc_before_pool, self.concat)

        g.ndata["h"] = g_embedding # [21352, 32]

        readout = dgl.sum_nodes(g, "h") # [64, 32]
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, "h")
            out_all.append(readout)

        adj, h = self.first_diffpool_layer(g, g_embedding) # [3392, 3392] [3392, 64]
        node_per_pool_graph = int(adj.size()[0] / len(g.batch_num_nodes())) # 53 = 3392 / 64

        h, adj = batch2tensor(adj, h, node_per_pool_graph) # [64, 53, 64] [64, 53, 53]
        h = self.gcn_forward_tensorized(
            h, adj, self.gc_after_pool[0], self.concat
        ) # [64, 53, 32]
        readout = torch.sum(h, dim=1) # [64, 32]
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)

        for i, diffpool_layer in enumerate(self.diffpool_layers):
            h, adj = diffpool_layer(h, adj)
            h = self.gcn_forward_tensorized(
                h, adj, self.gc_after_pool[i + 1], self.concat
            )
            readout = torch.sum(h, dim=1)
            out_all.append(readout)
            if self.num_aggs == 2:
                readout, _ = torch.max(h, dim=1)
                out_all.append(readout)
        if self.concat or self.num_aggs > 1:
            final_readout = torch.cat(out_all, dim=1)
        else:
            final_readout = readout
        ypred = self.pred_layer(final_readout)
        return ypred

    def loss(self, pred, label):
        """
        loss function
        """
        # softmax + CE
        # graph_label_dict = {'UDP-Glc': 0, 'UDP-GlcNAc': 1, 'UDP-GlcA': 2,
        #             'UDP-Gal': 3, 'UDP-GalNAc': 4,
        #             'UDP-Xyl': 5, 'GDP-Man': 6, 'GDP-Fuc': 7,
        #             'dTDP-Rha': 8, 'Other': 9}
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 2.6, 17.6, 4, 16, 40, 4, 80, 20, 2.2]).cuda()) # 没数据增强-old
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 2.4, 20, 6.4, 10.4, 44.8, 3, 102.4, 23.2, 2.4]).cuda()) # 数据增强-old
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 23.0, 8.0, 17.0, 69.0, 4.0, 102.0, 29.0, 3.0]).cuda()) # 没数据增强-快速收敛测试
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(self.custom_loss_weight).cuda()) # 支持外部传输的损失权重
        loss = criterion(pred, label)
        for key, value in self.first_diffpool_layer.loss_log.items():
            loss += value
        for diffpool_layer in self.diffpool_layers:
            for key, value in diffpool_layer.loss_log.items():
                loss += value
        return loss
    


    '''
    想加focal loss来着，但是diffpool好像有动态生成什么，导致我用下面的代码后，loss接受的pred都变了。
    我暂时已经尽力了，无法完成focal loss的应用。这让我之后用钩子函数去获取显著性分析充满了失望。
    ————2024/12/19
    '''
    # def loss(self, pred, label):
    #     """
    #     softmax focal loss function
    #     """
    #     print('===================================loss===============================')
        
    #     # 基础数据
    #     eps=1e-7
    #     weight = torch.tensor([[4710/3699],
    #                            [4710/939],
    #                            [4710/72]]).cuda()
    #     pred = pred.view((pred.size()[0],pred.size()[1],-1))

    #     logits_min = pred.min(dim=0, keepdim=True)[0]
    #     logits_max = pred.max(dim=0, keepdim=True)[0]
    #     pred = (pred - logits_min) / (logits_max - logits_min)
    #     print(pred)
    #     pred = torch.softmax(pred, dim=1)
    #     print(pred)

    #     one_hot_labels = F.one_hot(label, num_classes=3).to(torch.float32)
    #     label = one_hot_labels
    #     # print(label)
    #     target = label.view(pred.size())

    #     ce=-1*torch.log(pred+eps)*target
    #     # print(ce)
    #     floss=torch.pow((1-pred),2)*ce
    #     floss=torch.mul(floss,weight)
    #     floss=torch.sum(floss,dim=1)
    #     floss = torch.mean(floss)
    #     loss = floss
        


    #     for key, value in self.first_diffpool_layer.loss_log.items():
    #         loss += value
    #     for diffpool_layer in self.diffpool_layers:
    #         for key, value in diffpool_layer.loss_log.items():
    #             loss += value
    #     return loss
