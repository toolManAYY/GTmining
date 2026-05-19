import argparse
import textwrap
import os
import time
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import tu
from model.encoder import DiffPool
from livelossplot import PlotLosses
from sklearn.metrics import f1_score
import shutil
import pandas as pd
import sys
import numpy as np



def prepare_data(dataset, shuffle=False, prog_args=None):
    """
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    """
    return dgl.dataloading.GraphDataLoader(
        dataset,
        batch_size=prog_args.batch_size,
        shuffle=shuffle,
        num_workers=prog_args.n_worker,
    )

class customreaddata:
    """
    A custom dataset class to read graph data from specified text files for prediction.
    Assumes the following files exist in raw_dir under a subdirectory named 'name':
    - {name}_A.txt (edge list)
    - {name}_graph_indicator.txt (which graph each node belongs to)
    - {name}_graph_labels.txt (labels for each graph) - This might be dummy for prediction
    - {name}_node_attributes.txt (features for each node)

    Parameters
    ----------
    name : str
        Name of the dataset directory and prefix for files (e.g., 'GTmining').
    raw_dir : str
        Path to the directory containing the dataset folder.
    """

    def __init__(self, name, raw_dir):
        self.name = name
        self.raw_dir = raw_dir
        self.save_dir = os.path.join(raw_dir, name) # Use raw_dir as base, create save path inside
        os.makedirs(self.save_dir, exist_ok=True) # Ensure save directory exists for potential caching

        # Initialize attributes that will be set by process()
        self.graph_lists = []
        self.graph_labels = []
        self.item_ids = []  # 新增：存储每个图的item_id
        self.max_num_node = 0
        self.num_labels = None # May not be relevant for prediction

        # Process the raw data files
        self.process()

    def _file_path(self, category):
        """Constructs the path to a specific data file."""
        return os.path.join(self.raw_dir, f"{self.name}_{category}.txt")

    @staticmethod
    def _idx_from_zero(idx_tensor):
        """Adjusts indices to be 0-based."""
        # Assuming node and graph indices in your files are 1-based.
        # If they are already 0-based, this step is unnecessary or needs adjustment.
        # Check the first few lines of your GTmining_graph_indicator.txt to confirm.
        # If they are 0-based, remove this function or make it a no-op.
        # For now, assuming 1-based as per TUDataset standard.
        min_val = np.min(idx_tensor)
        if min_val == 0:
             # If already 0-based, return as is or handle accordingly
             # This might be the case, adjust logic if needed
             print(f"Warning: Indices in file seem to be 0-based (min={min_val}). Proceeding assuming 0-based.")
             return idx_tensor
        else:
             # Standard 1-based to 0-based conversion
             return idx_tensor - 1 # More standard than subtracting min if known to start at 1

    def process(self):
        """
        Loads data from text files and constructs a list of DGLGraphs.
        """
        print(f"Processing custom dataset: {self.name}")

        # --- 1. Load Edge List ---
        print(f"Loading edges from {self._file_path('A')}")
        # Load edges, assuming 1-based indexing initially
        edge_data_raw = np.genfromtxt(self._file_path("A"), delimiter=",", dtype=int)
        if edge_data_raw.ndim == 1:
            # If only one edge, reshape to (1, 2)
            edge_data_raw = edge_data_raw.reshape(1, -1)
        # Convert to 0-based indices
        edge_data_0_based = self._idx_from_zero(edge_data_raw)
        # DGL expects source and destination arrays
        src_nodes = edge_data_0_based[:, 0]
        dst_nodes = edge_data_0_based[:, 1]

        # --- 2. Load Graph Indicator (which graph each node belongs to) ---
        print(f"Loading graph indicators from {self._file_path('graph_indicator')}")
        node_graph_ids_raw = np.loadtxt(self._file_path("graph_indicator"), dtype=int)
        # Convert graph IDs to 0-based indices
        node_graph_ids = self._idx_from_zero(node_graph_ids_raw)
        num_total_nodes_in_file = len(node_graph_ids_raw)

        # --- 3. Load Graph Labels (might be dummy) ---
        print(f"Loading graph labels from {self._file_path('graph_labels')}")
        try:
            graph_labels_raw = np.loadtxt(self._file_path("graph_labels"), dtype=int)
            # Convert graph labels to 0-based indices if needed, though for classification
            # they often represent class IDs starting from 0 or 1. Adjust if necessary.
            # For prediction, these might just be placeholders.
            self.graph_labels = graph_labels_raw # Keep original values for now, adjust if necessary
            self.num_labels = max(self.graph_labels) + 1 if len(self.graph_labels) > 0 else 0
        except FileNotFoundError:
            print(f"Warning: Graph labels file {self._file_path('graph_labels')} not found. Using dummy labels (e.g., 0).")
            num_graphs_in_file = len(set(node_graph_ids))
            self.graph_labels = np.zeros(num_graphs_in_file, dtype=int) # Dummy labels
            self.num_labels = 1 # Or set to None if not applicable

        # --- 4. Load Item IDs (新增逻辑) ---
        print(f"Loading item IDs from {self._file_path('itemID')}")
        try:
            # 读取itemID文件，支持整数或字符串类型的ID
            self.item_ids = np.loadtxt(self._file_path("itemID"), dtype=str).tolist()
            
            # 验证item_id数量是否与图数量匹配
            num_graphs_in_file = len(set(node_graph_ids))
            if len(self.item_ids) != num_graphs_in_file:
                raise ValueError(
                    f"Number of item IDs ({len(self.item_ids)}) does not match number of graphs ({num_graphs_in_file})."
                )
        except FileNotFoundError:
            print(f"Warning: Item ID file {self._file_path('itemID')} not found. Using graph index as item ID.")
            num_graphs_in_file = len(set(node_graph_ids))
            self.item_ids = [str(i) for i in range(num_graphs_in_file)]  # 使用索引作为默认ID


        # --- 4. Load Node Attributes ---
        print(f"Loading node attributes from {self._file_path('node_attributes')}")
        try:
            node_attributes = np.loadtxt(self._file_path("node_attributes"), delimiter=",")
            if node_attributes.ndim == 1:
                # If features are 1D (one feature per node), reshape to (num_nodes, 1)
                node_attributes = np.expand_dims(node_attributes, axis=1)
            print(f"Loaded node attributes with shape: {node_attributes.shape}")
            if node_attributes.shape[0] != num_total_nodes_in_file:
                 raise ValueError(f"Number of rows in node_attributes ({node_attributes.shape[0]}) does not match number of nodes indicated by graph_indicator ({num_total_nodes_in_file}).")
        except FileNotFoundError:
            print(f"Warning: Node attributes file {self._file_path('node_attributes')} not found. Graphs will have no node features (ndata['feat'] will not be set initially).")
            node_attributes = None


        # --- 5. Create a Base Graph with All Nodes and Edges ---
        # This graph contains all nodes from all graphs, connected by the provided edges.
        num_nodes_in_base_graph = int(np.max(src_nodes)) + 1 if len(src_nodes) > 0 else 0
        # Ensure num_nodes includes any isolated nodes that might only appear in graph_indicator
        num_nodes_in_base_graph = max(num_nodes_in_base_graph, num_total_nodes_in_file)

        if num_nodes_in_base_graph == 0:
            print("Warning: No nodes or edges found in the data files.")
            self.graph_lists = []
            return

        base_graph = dgl.graph(([], []), num_nodes=num_nodes_in_base_graph)
        base_graph.add_edges(src_nodes, dst_nodes)

        # Assign node attributes to the base graph if available
        if node_attributes is not None:
            base_graph.ndata['feat'] = torch.tensor(node_attributes, dtype=torch.float32)

        # --- 6. Split the Base Graph into Individual Graphs ---
        self.graph_lists = []
        self.max_num_node = 0

        num_expected_graphs = len(set(node_graph_ids))
        print(f"Found {num_expected_graphs} graphs based on graph_indicator.")

        for graph_id in range(num_expected_graphs):
            # Find the nodes belonging to the current graph (graph_id)
            node_mask = (node_graph_ids == graph_id)
            node_indices_for_graph = np.where(node_mask)[0] # Get 0-based indices of nodes in this graph

            if len(node_indices_for_graph) == 0:
                print(f"Warning: Graph ID {graph_id} has no nodes according to graph_indicator.")
                # Create an empty graph for this ID
                g_sub = dgl.graph(([], []), num_nodes=0)
                # Add a dummy feature tensor if original had features, though shape might be tricky for 0 nodes
                # Often, empty graphs might need special handling downstream.
                # For now, just create the empty graph.
            else:
                # Extract the subgraph corresponding to these nodes
                g_sub = base_graph.subgraph(node_indices_for_graph)

                # The subgraph's nodes have new IDs (0, 1, ...). The original features are preserved based on the subgraph operation.
                # If node_attributes was loaded, 'feat' is already in g_sub.ndata.
                # Check if 'feat' exists, otherwise features were not available.
                if 'feat' not in g_sub.ndata:
                     print(f"  Graph {graph_id}: No node features available.")


            self.graph_lists.append(g_sub)

            if g_sub.num_nodes() > self.max_num_node:
                self.max_num_node = g_sub.num_nodes()

        print(f"Successfully processed {len(self.graph_lists)} graphs.")
        print(f"Max number of nodes in a single graph: {self.max_num_node}")


    def __getitem__(self, idx):
        """
        Gets the graph and its label at the given index.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        dgl.DGLGraph
            The graph object, potentially with node features in `ndata['feat']`.
        torch.Tensor
            The label tensor for the graph (could be dummy for prediction).
        """
        if idx < 0 or idx >= len(self):
             raise IndexError(f"Index {idx} is out of range for dataset with {len(self)} items.")
        g = self.graph_lists[idx]
        label = torch.tensor(self.graph_labels[idx], dtype=torch.int64) if self.graph_labels is not None else torch.tensor(0, dtype=torch.int64) # Return a dummy label if not set
        item_id = self.item_ids[idx]  # 新增：获取对应索引的item_id

        return g, label, item_id

    def __len__(self):
        """
        Returns the number of graphs in the dataset.
        """
        return len(self.graph_lists)

    @property
    def num_classes(self):
        """Returns the number of classes (uses num_labels)."""
        return int(self.num_labels) if self.num_labels is not None else 0 # Return 0 if not set
    
    def statistics(self):
        """
        返回数据集的三个关键统计信息，适配你的调用格式：
        input_dim, label_dim, max_num_node
        """
        # 1. 节点特征维度（如果没有特征则返回 0）
        if len(self.graph_lists) > 0 and 'feat' in self.graph_lists[0].ndata:
            input_dim = self.graph_lists[0].ndata['feat'].shape[1]
        else:
            input_dim = 0

        # 2. 标签维度（类别数量）
        label_dim = self.num_classes

        # 3. 最大节点数（你已经在 process 里算好了）
        max_num_node = self.max_num_node

        return input_dim, label_dim, max_num_node




def train(dataset, model, prog_args, val_dataset=None):
    """
    training function
    """
    # 初始化模型存储路径
    if not os.path.exists(prog_args.save_dir + "/" + prog_args.dataset):
        os.makedirs(prog_args.save_dir + "/" + prog_args.dataset)
    else:
        shutil.rmtree(prog_args.save_dir + "/" + prog_args.dataset)
        os.makedirs(prog_args.save_dir + "/" + prog_args.dataset)

    f_train_log = open(prog_args.save_dir + "/" + prog_args.dataset + "/train_log.csv", 'w')
    temp_line = 'epoch, accuracy, F1 score'
    line_flag = 3
    if prog_args.family_fold_type == 'GTA':
        for x in range(1, 10):
            temp_line = temp_line + f', Class {x} F1 score'
            line_flag += 1
    elif prog_args.family_fold_type == 'GTB':
        for x in range(1, 11):
            temp_line = temp_line + f', Class {x} F1 score'
            line_flag += 1
    else:
        raise ValueError(f"Invalid family_fold_type: '{prog_args.family_fold_type}'. Valid options are 'GTA' and 'GTB'.")
    temp_line = temp_line + '\n'
    f_train_log.write(temp_line)

    # 检查存储路径
    dir = prog_args.save_dir + "/" + prog_args.dataset
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    dataloader = dataset
    optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))  # 初始化优化器

    if prog_args.cuda > 0:
        torch.cuda.set_device(0)
    
    for epoch in range(0, prog_args.epoch):
        # 暂时存储，方便计算准确性
        train_pred_indi = torch.tensor([], device='cuda')
        train_label_indi = torch.tensor([], device='cuda')
        begin_time = time.time()
        model.train()
        accum_correct = 0
        total = 0
        print("\nEPOCH ###### {} ######".format(epoch))
        computation_time = 0.0
        for batch_idx, (batch_graph, graph_labels, item_ids) in enumerate(dataloader):
            for key, value in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            if torch.cuda.is_available():
                batch_graph = batch_graph.to(torch.cuda.current_device())
                graph_labels = graph_labels.cuda()

            model.zero_grad()
            compute_start = time.time()
            ypred = model(batch_graph)
            indi = torch.argmax(ypred, dim=1)
            train_pred_indi = torch.cat((train_pred_indi, indi), dim=0)
            train_label_indi = torch.cat((train_label_indi, graph_labels), dim=0)
            correct = torch.sum(indi == graph_labels).item()
            accum_correct += correct
            total += graph_labels.size()[0]
            loss = model.loss(ypred, graph_labels)
            loss.backward()
            batch_compute_time = time.time() - compute_start
            computation_time += batch_compute_time
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=prog_args.clip)
            optimizer.step()
        
        train_f1_score = f1_score(train_pred_indi.cpu(), train_label_indi.cpu(), average='macro')
        temp_f1_score = f1_score(train_pred_indi.cpu(), train_label_indi.cpu(), average=None)
        train_f1_score_class_dict = {}
        train_f1_score_class_keys = []
        for i in range(len(temp_f1_score)):
            # 生成与原变量名相同的key，如"train_f1_score_class_1"
            key = f"train_f1_score_class_{i+1}"
            # 存储对应的值（i是列表索引，i+1是类别编号）
            train_f1_score_class_keys.append(key) # 列表是有先后顺序的，保证写入log的时候不会乱序
            train_f1_score_class_dict[key] = temp_f1_score[i]
        train_accu = accum_correct / total

        elapsed_time = time.time() - begin_time

        if prog_args.save_dir is not None:
            torch.save(
                model.state_dict(),
                prog_args.save_dir
                + "/"
                + prog_args.dataset
                + "/model.iter-"
                + "{:04d}".format(epoch)
            )

        temp_line = f'{epoch}, {train_accu * 100}, {train_f1_score}'
        temp_line_flag = 3
        for x in train_f1_score_class_keys:
            temp_line = temp_line + f', {train_f1_score_class_dict[x]}'
            temp_line_flag += 1
        temp_line = temp_line + '\n'
        assert temp_line_flag == line_flag, 'Wrong log line number, please check what happen.'
        f_train_log.write(temp_line)

        print("train accuracy for this epoch {} is {:.2f}%".format(epoch, train_accu * 100))
        print("loss {:.4f} with epoch time {:.4f} s & computation time {:.4f} s ".format(loss.item(), elapsed_time, computation_time))

        torch.cuda.empty_cache()
    f_train_log.close()
    return 'Trian successfully'



# fold_num = 1
# family_fold_type = 'GTA'
# 获取命令行参数
args = sys.argv[1:]
fold_num = int(args[0].strip())
family_fold_type = str(args[1].strip())
abl_feature = str(args[2].strip())

print("{:=^100}".format(f'fold num is : {fold_num}, family type is : {family_fold_type}'))

print("{:=^100}".format('prog_args'))
prog_args = argparse.Namespace(dataset=f'GTmining_6_6_{family_fold_type}_fold{fold_num}_{abl_feature}', pool_ratio=0.10, num_pool=1, cuda=1, lr=1.0, clip=float("inf"),
                               batch_size=128, epoch=1000, n_worker=10, gc_per_block=3, aggregator_type="meanpool",
                               dropout=0.00, method="diffpool", bn=True, bias=True, save_dir=f"./model_param_alldata_abl",
                               load_epoch=-1, data_mode="default", linkpred=False, hidden_dim=64, embedding_dim=64, family_fold_type=family_fold_type)
print( textwrap.fill(str(prog_args), width=100))

print("{:=^100}".format('加载数据'))
dataset_train = customreaddata(name="GTmining",
                                    raw_dir=f'../data/dl_data/{family_fold_type}_alldata_id_abl_{abl_feature}/fold{fold_num}/train/')
dataset_validation = customreaddata(name="GTmining",
                                   raw_dir=f'../data/dl_data/{family_fold_type}_alldata_id_abl_{abl_feature}/fold{fold_num}/validation/')
dataset_test = customreaddata(name="GTmining",
                                   raw_dir=f'../data/dl_data/{family_fold_type}_alldata_id_abl_{abl_feature}/fold{fold_num}/test/')
train_dataloader = prepare_data(dataset_train, shuffle=True, prog_args=prog_args)
validation_dataloader = prepare_data(dataset_validation, shuffle=False, prog_args=prog_args)
test_dataloader = prepare_data(dataset_test, shuffle=False, prog_args=prog_args)


input_dim_train, label_dim_train, max_num_node_train = dataset_train.statistics()
input_dim_validation, label_dim_validation, max_num_node_validation = dataset_validation.statistics()
input_dim_test, label_dim_test, max_num_node_test = dataset_test.statistics()
max_num_node = max([max_num_node_train, max_num_node_validation, max_num_node_test])
input_dim = input_dim_train
label_dim = label_dim_train
print("++++++++++ STATISTICS ABOUT THE DATASET ++++++++++")
print("dataset feature dimension is", input_dim_train)
print("dataset label dimension is", label_dim_train)
print("the max num node is", max_num_node)
print("number of graphs is", len(dataset_train) + len(dataset_validation)+ len(dataset_test))

hidden_dim = prog_args.hidden_dim  # used to be 64
embedding_dim = prog_args.embedding_dim

assign_dim = int(max_num_node * prog_args.pool_ratio)
print("++++++++++MODEL STATISTICS++++++++")
print("model hidden dim is", hidden_dim)
print("model embedding dim for graph instance embedding", embedding_dim)
print("initial batched pool graph dim is", assign_dim)
activation = F.relu

if family_fold_type == 'GTA':
    graph_label_dict = {'UDP-Glc': 0, 'UDP-GlcNAc': 1, 'UDP-GlcA': 2,
                        'UDP-Gal': 3, 'UDP-GalNAc': 4,
                        'UDP-Xyl': 5, 'GDP-Man': 6,
                        'dTDP-Rha': 7, 'Other': 8}
elif family_fold_type == 'GTB':
    graph_label_dict = {'UDP-Glc': 0, 'UDP-GlcNAc': 1, 'UDP-GlcA': 2,
                        'UDP-Gal': 3, 'UDP-GalNAc': 4,
                        'UDP-Xyl': 5, 'GDP-Man': 6, 'GDP-Fuc': 7,
                        'dTDP-Rha': 8, 'Other': 9}
else:
    raise ValueError(f"Invalid family_fold_type: '{prog_args.family_fold_type}'. Valid options are 'GTA' and 'GTB'.")
df_cluster = pd.read_excel(f'../data/cluster/{family_fold_type}_alldata/dataseat_split_{fold_num}.xlsx')
df_cluster = df_cluster.loc[df_cluster['Dataset']=='train']
df_cluster.reset_index(drop=True, inplace=True)
custom_loss_weight = []
total_sample = df_cluster.shape[0]
for x in graph_label_dict.keys():
    df_x = df_cluster.loc[df_cluster['Activate']==x]
    df_x.reset_index(drop=True, inplace=True)
    x_sample = df_x.shape[0]
    custom_loss_weight.append(total_sample/x_sample)

assert len(custom_loss_weight) == label_dim_train, 'Wrong custom loss weight, please check what happen.'

# initialize model
model = DiffPool(
    input_dim,
    hidden_dim,
    embedding_dim,
    label_dim,
    activation,
    prog_args.gc_per_block,
    prog_args.dropout,
    prog_args.num_pool,
    prog_args.linkpred,
    prog_args.batch_size,
    prog_args.aggregator_type,
    assign_dim,
    prog_args.pool_ratio,
    custom_loss_weight
)
print("model init finished")
print("MODEL:::::::", prog_args.method)
if prog_args.cuda:
    model = model.cuda()


logger = train(
    train_dataloader, model, prog_args, val_dataset=test_dataloader
)