import torch
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, AvgPooling, MaxPooling, GATConv,SumPooling,SAGEConv,ChebConv
from model.utils import topk, get_batch_id

class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper 
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """
    def __init__(self, in_dim:int, ratio=0.5, conv_op=GraphConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer1 = GraphConv(in_dim, 1)
        self.score_layer2 = GraphConv(in_dim, 1)
        self.non_linearity = non_linearity
        self.allow_zero_in_degree = True 
    
    def forward(self, graph:dgl.DGLGraph, feature:torch.Tensor):

        score1 = self.score_layer1(graph, feature).squeeze()
        score2 = self.score_layer2(graph, feature).squeeze()
        score  = (score1+score2)/2
        perm, next_batch_num_nodes = topk(score, self.ratio, get_batch_id(graph.batch_num_nodes()), graph.batch_num_nodes())
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        graph = dgl.node_subgraph(graph, perm)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)
        
        return graph, feature, perm

#因为加入了CBAM模块，所以需要修改原有的ConvPoolBlock
#以下为原有的
# class ConvPoolBlock(torch.nn.Module):
#     """A combination of GCN layer and SAGPool layer,
#     followed by a concatenated (max||sum) readout operation.
#     """
#     def __init__(self, in_dim:int, out_dim:int, pool_ratio=0.5):
#         super(ConvPoolBlock, self).__init__()
#         self.conv1 = GraphConv(in_dim, out_dim)
#         self.conv2 = GraphConv(out_dim, out_dim)
#         self.pool = SAGPool(out_dim, ratio=pool_ratio)
#         self.avgpool = AvgPooling()
#         self.maxpool = MaxPooling()
#         self.sumpool = SumPooling()
#         self.allow_zero_in_degree = True   
    
#     def forward(self, graph, feature):
#         out = F.relu(self.conv1(graph, feature))
#         out = torch.reshape(out,(-1,512))
#         out = F.relu(self.conv2(graph, out))
#         out = torch.reshape(out,(-1,512))
#         out = F.relu(self.conv2(graph, out))
#         out = torch.reshape(out,(-1,512))
#         graph, out, _ = self.pool(graph, out)
#         g_out = torch.cat([self.maxpool(graph, out), self.sumpool(graph, out)], dim=-1)
#         return graph, out, g_out 


#新内容
#尝试添加CBAM模块
class GraphChannelAttention(torch.nn.Module):
    """图节点特征的通道注意力 - 用于特征增强"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(GraphChannelAttention, self).__init__()
        self.avg_pool = dgl.nn.AvgPooling()
        self.max_pool = dgl.nn.MaxPooling()
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels // reduction_ratio),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, graph, x):
        # 计算图级别的特征统计
        avg_pool = self.avg_pool(graph, x)
        max_pool = self.max_pool(graph, x)
        
        # 生成通道注意力权重
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        channel_att = torch.sigmoid(avg_out + max_out)
        
        # 将注意力权重扩展到所有节点
        batch_size = graph.batch_size if hasattr(graph, 'batch_size') else 1
        if batch_size > 1:
            num_nodes = graph.batch_num_nodes()
            node_channel_att = []
            for i in range(batch_size):
                node_att = channel_att[i].unsqueeze(0).repeat(num_nodes[i], 1)
                node_channel_att.append(node_att)
            node_channel_att = torch.cat(node_channel_att, dim=0)
        else:
            node_channel_att = channel_att.repeat(x.size(0), 1)
        
        return x * node_channel_att

class GraphSpatialAttention(torch.nn.Module):
    """图节点的空间注意力 - 用于节点重要性建模"""
    def __init__(self):
        super(GraphSpatialAttention, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )
        
    def forward(self, graph, x):
        # 对每个节点，计算特征维度上的统计信息
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [num_nodes, 1]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [num_nodes, 1]
        
        # 拼接并通过MLP生成空间注意力权重
        concat = torch.cat([avg_pool, max_pool], dim=1)  # [num_nodes, 2]
        spatial_att = torch.sigmoid(self.mlp(concat))  # [num_nodes, 1]
        
        return x * spatial_att

class GraphCBAM(torch.nn.Module):
    """适用于图神经网络的CBAM - 作为特征增强模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(GraphCBAM, self).__init__()
        self.channel_attention = GraphChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = GraphSpatialAttention()
        
    def forward(self, graph, x):
        # 依次应用通道注意力和空间注意力
        x = self.channel_attention(graph, x)
        x = self.spatial_attention(graph, x)
        return x

# 新的ConvPoolBlock

# class ConvPoolBlock(torch.nn.Module):
#     """A combination of GCN layer and SAGPool layer,
#     followed by a concatenated (max||sum) readout operation.
#     现在增加了可选的Graph-CBAM特征增强
#     """
#     def __init__(self, in_dim:int, out_dim:int, pool_ratio=0.5, use_cbam=False):
#         super(ConvPoolBlock, self).__init__()
#         self.conv1 = GraphConv(in_dim, out_dim)
#         self.conv2 = GraphConv(out_dim, out_dim)
#         self.pool = SAGPool(out_dim, ratio=pool_ratio)  # 保留原有的SAGPool
#         self.avgpool = AvgPooling()
#         self.maxpool = MaxPooling()
#         self.sumpool = SumPooling()
        
#         # 新增：可选的Graph-CBAM模块
#         self.use_cbam = use_cbam
#         if use_cbam:
#             self.cbam = GraphCBAM(out_dim)
        
#         self.allow_zero_in_degree = True   
    
#     def forward(self, graph, feature):
#         # 原有的图卷积操作
#         out = F.relu(self.conv1(graph, feature))
#         out = torch.reshape(out,(-1,512))
#         out = F.relu(self.conv2(graph, out))
#         out = torch.reshape(out,(-1,512))
        
#         # 新增：在SAGPool之前应用CBAM进行特征增强
#         if self.use_cbam:
#             out = self.cbam(graph, out)
        
#         # 原有的第三层卷积和SAGPool（保持不变）
#         out = F.relu(self.conv2(graph, out))
#         out = torch.reshape(out,(-1,512))
#         graph, out, _ = self.pool(graph, out)  # 原有的自注意力池化
        
#         # 原有的readout操作
#         g_out = torch.cat([self.maxpool(graph, out), self.sumpool(graph, out)], dim=-1)
#         return graph, out, g_out

class ConvPoolBlock(torch.nn.Module):
    """A combination of GCN layer and SAGPool layer,
    followed by a concatenated (max||sum) readout operation.
    现在增加了可选的Graph-CBAM特征增强
    """
    def __init__(self, in_dim:int, out_dim:int, pool_ratio=0.5, use_cbam=False):
        super(ConvPoolBlock, self).__init__()
        self.conv1 = GraphConv(in_dim, out_dim)
        self.conv2 = GraphConv(out_dim, out_dim)
        self.pool = SAGPool(out_dim, ratio=pool_ratio)  # 保留原有的SAGPool
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        self.sumpool = SumPooling()
        
        # 新增：可选的Graph-CBAM模块
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = GraphCBAM(out_dim)
        
        self.allow_zero_in_degree = True   
    
    def forward(self, graph, feature):
        # print("graph batch size point 1:", graph.batch_size)
        # 原有的图卷积操作
        out = F.relu(self.conv1(graph, feature))
        # out = torch.reshape(out,(-1,512))
        out = F.relu(self.conv2(graph, out))
        # out = torch.reshape(out,(-1,512))
        # print("conv output shape:", out.shape) # [21984, 512]

        # 新增：在SAGPool之前应用CBAM进行特征增强
        if self.use_cbam:
            out = self.cbam(graph, out)
            # print("after CBAM shape:", out.shape) # [21984, 512]
        
        # # 原有的第三层卷积和SAGPool（保持不变）
        # out = F.relu(self.conv2(graph, out))
        # out = torch.reshape(out,(-1,512))
        graph, out, _ = self.pool(graph, out)  # 原有的自注意力池化
        # print("graph batch size point 2:", graph.batch_size)
        # print("after pooling shape:", out.shape) # [16596, 512]

        
        # 原有的readout操作
        # print("graph batch size:", graph.batch_size) #
        # print("maxpool shape:", self.maxpool(graph, out).shape) # [1, 512]
        # print("sumpool shape:", self.sumpool(graph, out).shape) # [1, 512]
        g_out = torch.cat([self.maxpool(graph, out), self.sumpool(graph, out)], dim=-1)
        # print("g_out shape:", g_out.shape) # [1, 1024]

        return graph, out, g_out
