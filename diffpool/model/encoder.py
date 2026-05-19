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

        self.g_embedding = g_embedding # 【必须注册为实例属性，用于梯度计算】显著性图修改

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

        # 【必须注册为实例属性，用于梯度计算】显著性图修改
        self.final_graph_embedding = final_readout

        ypred = self.pred_layer(final_readout)
        return ypred

    #     # ===================== 【核心新增：Grad-CAM 显著性提取】 =====================
    # @torch.no_grad()
    # def get_gradient_cam(self, g, target_label=None):
    #     """
    #     提取 输入图 的节点显著性（Grad-CAM）
    #     :param g: 输入 DGL 图（batch 或单张图）
    #     :param target_label: 目标类别（不指定则使用模型预测的类别）
    #     :return: node_saliency: 每个节点的显著性分数 [N,]
    #     """
    #     self.eval()
    #     g_cp = g.clone().to(next(self.parameters()).device)
        
    #     # 1. 前向传播，计算预测值
    #     with torch.enable_grad():
    #         # 前向传播
    #         pred = self.forward(g_cp)

    #         # 确定目标标签
    #         if target_label is None:
    #             target_label = pred.argmax(dim=1)
    #         else:
    #             if not isinstance(target_label, torch.Tensor):
    #                 target_label = torch.tensor([target_label] * pred.shape[0], device=pred.device)
            
    #         # 2. 计算 目标类别 对 最终图嵌入 的梯度
    #         score = pred[torch.arange(pred.size(0), device=pred.device), target_label]
    #         grad = torch.autograd.grad(
    #             outputs=score.sum(),
    #             inputs=self.final_graph_embedding,
    #             retain_graph=True
    #         )[0]  # [batch_size, emb_dim]

    #     # 3. 重新计算原始图节点嵌入（第一层GNN输出，对应输入图节点）
    #     h = g_cp.ndata["feat"]
    #     node_emb = self.gcn_forward(g_cp, h, self.gc_before_pool, self.concat)  # [N, emb_dim]

    #     # 4. 梯度权重 × 节点嵌入 → 节点显著性
    #     batch_n_nodes = g_cp.batch_num_nodes()
    #     node_saliency = []
    #     ptr = 0
        
    #     for b in range(len(batch_n_nodes)):
    #         n_node = batch_n_nodes[b]
    #         g_weight = grad[b:b+1]  # [1, emb_dim] # 第 b 张图的梯度权重
    #         node_emb_batch = node_emb[ptr:ptr+n_node]  # [n_node, emb_dim]
            
    #         # 加权求和 → 显著性
    #         sal = (node_emb_batch * g_weight).sum(dim=1)
    #         node_saliency.append(sal)
    #         ptr += n_node
        
    #     # 5. 归一化到 [0,1]，便于可视化
    #     node_saliency = torch.cat(node_saliency)
    #     node_saliency = (node_saliency - node_saliency.min()) / (node_saliency.max() - node_saliency.min() + 1e-8)
        
    #     return node_saliency
    # # ==========================================================================

    
    # ===================== 【核心修改：Grad-CAM 显著性提取】 =====================
    def get_gradient_cam(self, g, target_label=None, pred=None, mode='node'):
        """
        提取 输入图 的节点显著性（Grad-CAM）
        :param g: 输入 DGL 图
        :param target_label: 目标类别
        :param pred: [可选] 外部传入的预测值（仅用于确定 target_label，前向传播会重新执行以确保计算图正确）
        :param mode: 显著性模式
                     - 'node': 输出每个节点的整体显著性分数 [N,] (原有逻辑，对最终图嵌入求导)
                     - 'node_feature': 输出节点-特征二维显著性图 [N, F] (对第一层嵌入求导)
        :return: node_saliency: 显著性分数 [N,] 或 [N, F]
        """
        self.eval()
        device = next(self.parameters()).device
        g_cp = g.clone().to(device)


        # 获取 Batch 信息备用
        batch_num_nodes = g_cp.batch_num_nodes() # e.g. [334, 335, ...] (Batch_size 个元素)
        total_num_nodes = sum(batch_num_nodes)    # e.g. 21352

        # ---------------------------------------------------------
        # 分支 1：原有逻辑 (mode='node')
        # 利用 self.final_graph_embedding 和 self.g_embedding
        # ---------------------------------------------------------
        if mode == 'node':
            with torch.enable_grad():
                # 1. 前向传播
                # 这一步会自动填充:
                #   self.g_embedding:          [Total_Nodes, 64]
                #   self.final_graph_embedding: [Batch_Size, Emb_Dim]
                pred_internal = self.forward(g_cp) # Shape: [Batch_Size, Num_Classes]

                # 2. 确定目标标签
                pred_to_use = pred if pred is not None else pred_internal
                if target_label is None:
                    target_label = pred_to_use.argmax(dim=1) # Shape: [Batch_Size]
                else:
                    if not isinstance(target_label, torch.Tensor):
                        target_label = torch.tensor([target_label] * pred_to_use.shape[0], device=device)
                
                # 3. 取出目标分数
                # 从 [Batch_Size, Classes] 中取出 [Batch_Size]
                score = pred_internal[torch.arange(pred_internal.size(0), device=device), target_label]
                
                # 4. 计算梯度
                # outputs: score.sum() (标量)
                # inputs: self.final_graph_embedding ([Batch_Size, Emb_Dim])
                # grad 形状: [Batch_Size, Emb_Dim]
                # print("The shape of final_graph_embedding:", self.final_graph_embedding.shape) # Debug 输出，确认形状是 [Batch_Size, Emb_Dim]
                # grad = torch.autograd.grad(
                #     outputs=score.sum(),
                #     inputs=self.final_graph_embedding,
                #     retain_graph=True
                # )[0]
                # print("The shape of g_embedding before grad:", self.g_embedding.shape) # Debug 输出，确认形状是 [Total_Nodes, Emb_Dim]
                grad = torch.autograd.grad(
                    outputs=score.sum(),
                    inputs=self.g_embedding,
                    retain_graph=True
                )[0]
                # node_saliency = grad.sum(dim=1)  # [Total_Nodes,] # 敏感度分数 = 梯度的绝对值（或直接求和）
                node_saliency = (grad * self.g_embedding).sum(dim=1)  # [Total_Nodes,] # 贡献度分数 = 梯度权重 × 原始嵌入（Grad-CAM 经典公式）
                # print("The shape of node_saliency before abs:", node_saliency.shape) # Debug 输出，确认形状是 [Total_Nodes,]
            
            # 只显示贡献大小（取绝对值），归一化到 [0, 1]
            # print("Max saliency before abs:", node_saliency.max().item(), "Min saliency before abs:", node_saliency.min().item()) # Debug 输出，确认数值范围
            node_saliency = torch.abs(node_saliency)
            node_saliency = (node_saliency - node_saliency.min()) / (node_saliency.max() - node_saliency.min() + 1e-8)

            return node_saliency



            # # 5. 取出之前保存的节点嵌入 (利用 forward 里的注册属性)
            # # node_emb 形状: [Total_Nodes, 64]
            # node_emb = self.g_embedding.detach()
            
            # # 6. 将 Batch 级别的梯度分配回每个节点
            # node_saliency = []
            # ptr = 0
            
            # for b in range(len(batch_num_nodes)):
            #     n_node = batch_num_nodes[b]
                
            #     # 取出第 b 张图的梯度: [1, Emb_Dim]
            #     g_weight = grad[b:b+1]
            #     print("The shape of g_weight:", g_weight.shape) # Debug 输出，确认形状是 [1, Emb_Dim]
                
            #     # 取出第 b 张图的节点: [n_node, Emb_Dim]
            #     node_emb_batch = node_emb[ptr:ptr+n_node]
                
            #     # 逐元素相乘后求和: [n_node, Emb_Dim] -> [n_node]
            #     sal = (node_emb_batch * g_weight).sum(dim=1)
            #     # sal = g_weight.sum(dim=1) --------------------


                
            #     node_saliency.append(sal)
            #     ptr += n_node
            

            
            # # 1. 先取绝对值，获取显著性强度
            # print("Max saliency before abs:", torch.cat(node_saliency).max().item(), "Min saliency before abs:", torch.cat(node_saliency).min().item())
            # node_saliency = torch.cat(node_saliency)  # Shape: [Total_Nodes]
            # node_saliency = torch.abs(node_saliency)

            # # 2. 再做 min-max 归一化到 [0,1]
            # node_saliency = (node_saliency - node_saliency.min()) / (node_saliency.max() - node_saliency.min() + 1e-8)

            # # # 拼接并归一化
            # # node_saliency = torch.cat(node_saliency) # Shape: [Total_Nodes]
            # # node_saliency = (node_saliency - node_saliency.min()) / (node_saliency.max() - node_saliency.min() + 1e-8)
            # return node_saliency

        # ---------------------------------------------------------
        # 分支 2：新逻辑 (mode='node_feature')
        # 针对原始输入 7 维特征
        # ---------------------------------------------------------
        else: # mode == 'node_feature'
            with torch.enable_grad():
                # 1. 获取原始输入特征
                # h 形状: [Total_Nodes, 7] (7 = x,y,z,si,hbond,charge,hphob)
                h = g_cp.ndata["feat"]
                h.requires_grad_(True) # 必须开启，才能对输入求导

                # 【关键技巧】把图里的特征替换成我们开启了梯度的这个 h
                # 这样后续调用 forward() 时，用的就是这个带梯度的 h 了
                g_cp.ndata["feat"] = h

                # 2. 直接调用 forward()！
                # 因为我们已经把 g_cp.ndata["feat"] 换成了 requires_grad=True 的版本
                # 所以整个计算图会自动连接到 h
                pred_internal = self.forward(g_cp)

                # 3. 确定 Label
                pred_to_use = pred if pred is not None else pred_internal
                if target_label is None:
                    target_label = pred_to_use.argmax(dim=1)
                else:
                    if not isinstance(target_label, torch.Tensor):
                        target_label = torch.tensor([target_label] * pred_to_use.shape[0], device=device)
                
                # 4. 计算 Score
                score = pred_internal[torch.arange(pred_internal.size(0), device=device), target_label]
                
                # 5. 【核心】计算梯度
                # inputs=h: 我们直接对原始的 7 维特征求导
                # grad 形状: [Total_Nodes, 7] (和 h 一模一样)
                grad = torch.autograd.grad(
                    outputs=score.sum(),
                    inputs=h,
                    retain_graph=True
                )[0]

            # 6. 计算显著性
            # h.detach(): [Total_Nodes, 7]
            # grad:       [Total_Nodes, 7]
            # 逐元素相乘:  [Total_Nodes, 7]
            node_saliency = h.detach() * grad
            
            # 返回原始的 [N, 7] 矩阵，不归一化，方便外部处理 (取绝对值、聚合XYZ等)
            return node_saliency





        
        # # ---------------------------------------------------------
        # # 1. 前向传播（必须执行，用于填充 self.g_embedding 和 self.final_graph_embedding）
        # # ---------------------------------------------------------
        # with torch.enable_grad():
        #     # 无论是否传入 pred，都重新跑一遍 forward，确保中间变量被正确记录且计算图连通
        #     pred_internal = self.forward(g_cp)

        #     # 确定目标标签（优先使用外部传入的 pred 来选 label，否则用内部算的）
        #     pred_to_use = pred if pred is not None else pred_internal
        #     if target_label is None:
        #         target_label = pred_to_use.argmax(dim=1)
        #     else:
        #         if not isinstance(target_label, torch.Tensor):
        #             target_label = torch.tensor([target_label] * pred_to_use.shape[0], device=pred_to_use.device)
            
        #     # 取出目标分数（使用内部重新计算的 pred，因为它连着计算图）
        #     score = pred_internal[torch.arange(pred_internal.size(0), device=pred_internal.device), target_label]
            
        #     # ---------------------------------------------------------
        #     # 2. 根据 mode 选择求导目标
        #     # ---------------------------------------------------------
        #     if mode == 'node':
        #         # 模式 A：原有逻辑，对【最终图嵌入】求导
        #         target_tensor = self.final_graph_embedding
        #     else: # mode == 'node_feature'
        #         # 模式 B：新逻辑，对【第一层节点嵌入】求导
        #         target_tensor = self.g_embedding
            
        #     # 计算梯度
        #     grad = torch.autograd.grad(
        #         outputs=score.sum(),
        #         inputs=target_tensor,
        #         retain_graph=True
        #     )[0]

        # # ---------------------------------------------------------
        # # 3. 计算显著性
        # # ---------------------------------------------------------
        # # 取出节点嵌入（用于相乘）
        # node_emb = self.g_embedding.detach()
        
        # if mode == 'node':
        #     # 原有逻辑：梯度是 [Batch, Emb_dim]，需要按 batch 分配回节点
        #     batch_n_nodes = g_cp.batch_num_nodes()
        #     node_saliency = []
        #     ptr = 0
            
        #     for b in range(len(batch_n_nodes)):
        #         n_node = batch_n_nodes[b]
        #         g_weight = grad[b:b+1]  # [1, emb_dim]
        #         node_emb_batch = node_emb[ptr:ptr+n_node]  # [n_node, emb_dim]
                
        #         sal = (node_emb_batch * g_weight).sum(dim=1)
        #         node_saliency.append(sal)
        #         ptr += n_node
            
        #     node_saliency = torch.cat(node_saliency)
            
        # else: # mode == 'node_feature'
        #     # 新逻辑：梯度已经是 [N, F]，直接逐元素相乘
        #     node_saliency = node_emb * grad

        # # ---------------------------------------------------------
        # # 4. 归一化
        # # ---------------------------------------------------------
        # if mode == 'node':
        #     # 全局归一化到 [0, 1]
        #     node_saliency = (node_saliency - node_saliency.min()) / (node_saliency.max() - node_saliency.min() + 1e-8)
        # else:
        #     # 按特征维度归一化（每一列单独缩放到 [0, 1]）
        #     min_vals = node_saliency.min(dim=0, keepdim=True)[0]
        #     max_vals = node_saliency.max(dim=0, keepdim=True)[0]
        #     # node_saliency = (node_saliency - min_vals) / (max_vals - min_vals + 1e-8)
        
        # return node_saliency









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
