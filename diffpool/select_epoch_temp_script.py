import os
import pandas as pd

best_100_epochs = {}
fold_type = 'GTA'
family_fold_type = fold_type
log_save_folder = f'GTmining_6_6_{fold_type}_fold'

for fold in range(1, 11):
    log_dir = f'./model_param_alldata/{log_save_folder}{fold}/validation_log.csv'
    df = pd.read_csv(log_dir, index_col=False)
    df.sort_values(by='validation_f1_score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    epochs_100 = df.loc[0:99, 'epoch'].values.tolist()
    best_100_epochs[fold] = epochs_100
    print(f"Fold type {fold_type}, Fold: {fold}, Top 100 Epochs: {epochs_100}")


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

import os
import pandas as pd

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

def prepare_data_one(dataset, shuffle=False, prog_args=None):
    """
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    """
    return dgl.dataloading.GraphDataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=prog_args.n_worker,
    )


fold_num = 1
# ========================================
data_path = "./model_param_alldata/"
base_name = f"GTmining_6_6_{family_fold_type}_fold"
folder_name = f"{base_name}{fold_num}"
# ========================================

print("{:=^100}".format(f'fold num is : {fold_num}, family type is : {family_fold_type}'))

print("{:=^100}".format('prog_args'))
prog_args = argparse.Namespace(dataset=f'GTmining_6_6_{family_fold_type}_fold{fold_num}', pool_ratio=0.10, num_pool=1, cuda=1, lr=1.0, clip=float("inf"),
                            batch_size=128, epoch=1000, n_worker=10, gc_per_block=3, aggregator_type="meanpool",
                            dropout=0.00, method="diffpool", bn=True, bias=True, save_dir=f"./model_param_alldata",
                            load_epoch=-1, data_mode="default", linkpred=False, hidden_dim=64, embedding_dim=64, family_fold_type=family_fold_type)
print(textwrap.fill(str(prog_args), width=100))

print("{:=^100}".format('加载数据'))
dataset_train = tu.LegacyTUDataset(name="GTmining",
                                    raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/train/')
dataset_validation = tu.LegacyTUDataset(name="GTmining",
                                raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/validation/')
dataset_test = tu.LegacyTUDataset(name="GTmining",
                                raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/test/')
train_dataloader = prepare_data(dataset_train, shuffle=True, prog_args=prog_args)
validation_dataloader = prepare_data_one(dataset_validation, shuffle=False, prog_args=prog_args)
test_dataloader = prepare_data_one(dataset_test, shuffle=False, prog_args=prog_args)

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

df_cluster = pd.read_excel(f'../data/cluster/{family_fold_type}/dataseat_split_{fold_num}.xlsx')
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

# 先过一遍train_dataloader，让模型中的一些参数先初始化一下
model.train()
for batch_idx, (batch_graph, graph_labels) in enumerate(train_dataloader):
    for key, value in batch_graph.ndata.items():
        batch_graph.ndata[key] = value.float()
    graph_labels = graph_labels.long()
    if torch.cuda.is_available():
        batch_graph = batch_graph.to(torch.cuda.current_device())
        graph_labels = graph_labels.cuda()
    ypred = model(batch_graph)
    loss = model.loss(ypred, graph_labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=prog_args.clip)
    model.zero_grad()


id_card_protein = {}
# validation test
with open(f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/validation/Predict_correspond_information.txt', 'r')as f:
    for dd in f.readlines():
        dd = dd.split('\n')[0].split('===')
        id_card_protein[dd[1]] = dd[0]



for fold_num in range(1, 11):
    # ========================================
    data_path = "./model_param_alldata/"
    base_name = f"GTmining_6_6_{family_fold_type}_fold"
    folder_name = f"{base_name}{fold_num}"
    # ========================================

    print("{:=^100}".format(f'fold num is : {fold_num}, family type is : {family_fold_type}'))

    print("{:=^100}".format('prog_args'))
    prog_args = argparse.Namespace(dataset=f'GTmining_6_6_{family_fold_type}_fold{fold_num}', pool_ratio=0.10, num_pool=1, cuda=1, lr=1.0, clip=float("inf"),
                                batch_size=128, epoch=1000, n_worker=10, gc_per_block=3, aggregator_type="meanpool",
                                dropout=0.00, method="diffpool", bn=True, bias=True, save_dir=f"./model_param_alldata",
                                load_epoch=-1, data_mode="default", linkpred=False, hidden_dim=64, embedding_dim=64, family_fold_type=family_fold_type)
    print(textwrap.fill(str(prog_args), width=100))

    print("{:=^100}".format('加载数据'))


    print("Fold_num is {}{:=^100}".format(fold_num, '开始选择最佳epoch进行验证'))
    for top_i in range(0, 100):
        epoch = best_100_epochs[fold_num][top_i]

        begin_time = time.time()
        print("\nEPOCH ###### {} ######".format(epoch))
        if epoch is not None and prog_args.save_dir is not None:
            model.load_state_dict(
                torch.load(
                    prog_args.save_dir
                    + "/"
                    + prog_args.dataset
                    + "/model.iter-"
                    + "{:04d}".format(epoch), weights_only=True
                )
            )
        

        id_card_record_list = []
        real_label_record_list = []
        predict_label_record_list = []
        graph_label_dict_reverse = {v: k for k, v in graph_label_dict.items()}

        model.eval()
        correct_label = 0
        with torch.no_grad():
            val_pred_indi = torch.tensor([], device='cuda')
            val_label_indi = torch.tensor([], device='cuda')
            for batch_idx, (batch_graph, graph_labels) in enumerate(validation_dataloader):
                for key, value in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.float()
                graph_labels = graph_labels.long()
                if torch.cuda.is_available():
                    batch_graph = batch_graph.to(torch.cuda.current_device())
                    graph_labels = graph_labels.cuda()

                # 拿标签
                temp = batch_graph
                protein_temp_id_card = ''
                protein_temp_id_card = protein_temp_id_card + "{:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}".format(round(float(temp.ndata['feat'][0][0]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][1]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][2]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][3]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][4]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][5]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][6]), 5))
                protein_temp_id_card = protein_temp_id_card + "###{:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}".format(round(float(temp.ndata['feat'][1][0]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][1]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][2]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][3]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][4]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][5]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][6]), 5))
                protein_temp_id_card = protein_temp_id_card + "###{:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}".format(round(float(temp.ndata['feat'][2][0]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][1]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][2]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][3]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][4]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][5]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][6]), 5))
                protein_id = id_card_protein[protein_temp_id_card]

                ypred = model(batch_graph)
                indi = torch.argmax(ypred, dim=1)

                # 记录id_card_record_list，real_label_record_list，predict_label_record_list
                id_card_record_list.append(protein_id)
                real_label_record_list.append(graph_label_dict_reverse[int(graph_labels.cpu())])
                predict_label_record_list.append(graph_label_dict_reverse[int(indi.cpu())])

                val_pred_indi = torch.cat((val_pred_indi, indi), dim=0)
                val_label_indi = torch.cat((val_label_indi, graph_labels), dim=0)
                correct = torch.sum(indi == graph_labels)
                correct_label += correct.item()

                # print(f"Protein name is : {protein_id}, and its activate is {int(indi.cpu())}.")
                
        elapsed_time = time.time() - begin_time
        print("epoch {:.4f} with epoch time {:.4f} s".format(epoch, elapsed_time))
        result = correct_label / len(validation_dataloader.dataset)

        # 记录预测的label数据文件
        storage_label_file = prog_args.save_dir + "/" + prog_args.dataset + f"/validation_pred_labels_epoch_{epoch}_top{top_i}.csv"

        with open(storage_label_file, 'w') as f:
            f.write("ID_Card,Real_Label,Predict_Label\n")
            for id_card, real_label, predict_label in zip(id_card_record_list, real_label_record_list, predict_label_record_list):
                f.write(f"{id_card},{real_label},{predict_label}\n")
        f.close()
    

























import os
import pandas as pd

best_50_epochs = {}
fold_type = 'GTB'
family_fold_type = fold_type
log_save_folder = f'GTmining_6_6_{fold_type}_fold'

for fold in range(1, 11):
    log_dir = f'./model_param_alldata/{log_save_folder}{fold}/validation_log.csv'
    df = pd.read_csv(log_dir, index_col=False)
    df.sort_values(by='validation_f1_score', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    epochs_50 = df.loc[0:49, 'epoch'].values.tolist()
    best_50_epochs[fold] = epochs_50
    print(f"Fold type {fold_type}, Fold: {fold}, Top 50 Epochs: {epochs_50}")


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

import os
import pandas as pd

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

def prepare_data_one(dataset, shuffle=False, prog_args=None):
    """
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    """
    return dgl.dataloading.GraphDataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=prog_args.n_worker,
    )


fold_num = 1
# ========================================
data_path = "./model_param_alldata/"
base_name = f"GTmining_6_6_{family_fold_type}_fold"
folder_name = f"{base_name}{fold_num}"
# ========================================

print("{:=^100}".format(f'fold num is : {fold_num}, family type is : {family_fold_type}'))

print("{:=^100}".format('prog_args'))
prog_args = argparse.Namespace(dataset=f'GTmining_6_6_{family_fold_type}_fold{fold_num}', pool_ratio=0.10, num_pool=1, cuda=1, lr=1.0, clip=float("inf"),
                            batch_size=128, epoch=1000, n_worker=10, gc_per_block=3, aggregator_type="meanpool",
                            dropout=0.00, method="diffpool", bn=True, bias=True, save_dir=f"./model_param_alldata",
                            load_epoch=-1, data_mode="default", linkpred=False, hidden_dim=64, embedding_dim=64, family_fold_type=family_fold_type)
print(textwrap.fill(str(prog_args), width=100))

print("{:=^100}".format('加载数据'))
dataset_train = tu.LegacyTUDataset(name="GTmining",
                                    raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/train/')
dataset_validation = tu.LegacyTUDataset(name="GTmining",
                                raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/validation/')
dataset_test = tu.LegacyTUDataset(name="GTmining",
                                raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/test/')
train_dataloader = prepare_data(dataset_train, shuffle=True, prog_args=prog_args)
validation_dataloader = prepare_data_one(dataset_validation, shuffle=False, prog_args=prog_args)
test_dataloader = prepare_data_one(dataset_test, shuffle=False, prog_args=prog_args)

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

df_cluster = pd.read_excel(f'../data/cluster/{family_fold_type}/dataseat_split_{fold_num}.xlsx')
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

# 先过一遍train_dataloader，让模型中的一些参数先初始化一下
model.train()
for batch_idx, (batch_graph, graph_labels) in enumerate(train_dataloader):
    for key, value in batch_graph.ndata.items():
        batch_graph.ndata[key] = value.float()
    graph_labels = graph_labels.long()
    if torch.cuda.is_available():
        batch_graph = batch_graph.to(torch.cuda.current_device())
        graph_labels = graph_labels.cuda()
    ypred = model(batch_graph)
    loss = model.loss(ypred, graph_labels)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=prog_args.clip)
    model.zero_grad()

id_card_protein = {}
# validation test
with open(f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/validation/Predict_correspond_information.txt', 'r')as f:
    for dd in f.readlines():
        dd = dd.split('\n')[0].split('===')
        id_card_protein[dd[1]] = dd[0]




for fold_num in range(1, 11):
    # ========================================
    data_path = "./model_param_alldata/"
    base_name = f"GTmining_6_6_{family_fold_type}_fold"
    folder_name = f"{base_name}{fold_num}"
    # ========================================

    print("{:=^100}".format(f'fold num is : {fold_num}, family type is : {family_fold_type}'))

    print("{:=^100}".format('prog_args'))
    prog_args = argparse.Namespace(dataset=f'GTmining_6_6_{family_fold_type}_fold{fold_num}', pool_ratio=0.10, num_pool=1, cuda=1, lr=1.0, clip=float("inf"),
                                batch_size=128, epoch=1000, n_worker=10, gc_per_block=3, aggregator_type="meanpool",
                                dropout=0.00, method="diffpool", bn=True, bias=True, save_dir=f"./model_param_alldata",
                                load_epoch=-1, data_mode="default", linkpred=False, hidden_dim=64, embedding_dim=64, family_fold_type=family_fold_type)
    print(textwrap.fill(str(prog_args), width=100))

    print("{:=^100}".format('加载数据'))

    print("Fold_num is {}{:=^100}".format(fold_num, '开始选择最佳epoch进行验证'))
    for top_i in range(0, 50):
        epoch = best_50_epochs[fold_num][top_i]

        begin_time = time.time()
        print("\nEPOCH ###### {} ######".format(epoch))
        if epoch is not None and prog_args.save_dir is not None:
            model.load_state_dict(
                torch.load(
                    prog_args.save_dir
                    + "/"
                    + prog_args.dataset
                    + "/model.iter-"
                    + "{:04d}".format(epoch), weights_only=True
                )
            )


        id_card_record_list = []
        real_label_record_list = []
        predict_label_record_list = []
        graph_label_dict_reverse = {v: k for k, v in graph_label_dict.items()}

        model.eval()
        correct_label = 0
        with torch.no_grad():
            val_pred_indi = torch.tensor([], device='cuda')
            val_label_indi = torch.tensor([], device='cuda')
            for batch_idx, (batch_graph, graph_labels) in enumerate(validation_dataloader):
                for key, value in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.float()
                graph_labels = graph_labels.long()
                if torch.cuda.is_available():
                    batch_graph = batch_graph.to(torch.cuda.current_device())
                    graph_labels = graph_labels.cuda()

                # 拿标签
                temp = batch_graph
                protein_temp_id_card = ''
                protein_temp_id_card = protein_temp_id_card + "{:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}".format(round(float(temp.ndata['feat'][0][0]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][1]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][2]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][3]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][4]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][5]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][0][6]), 5))
                protein_temp_id_card = protein_temp_id_card + "###{:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}".format(round(float(temp.ndata['feat'][1][0]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][1]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][2]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][3]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][4]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][5]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][1][6]), 5))
                protein_temp_id_card = protein_temp_id_card + "###{:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}".format(round(float(temp.ndata['feat'][2][0]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][1]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][2]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][3]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][4]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][5]), 5),
                                                                                                                                        round(float(temp.ndata['feat'][2][6]), 5))
                protein_id = id_card_protein[protein_temp_id_card]

                ypred = model(batch_graph)
                indi = torch.argmax(ypred, dim=1)

                # 记录id_card_record_list，real_label_record_list，predict_label_record_list
                id_card_record_list.append(protein_id)
                real_label_record_list.append(graph_label_dict_reverse[int(graph_labels.cpu())])
                predict_label_record_list.append(graph_label_dict_reverse[int(indi.cpu())])

                val_pred_indi = torch.cat((val_pred_indi, indi), dim=0)
                val_label_indi = torch.cat((val_label_indi, graph_labels), dim=0)
                correct = torch.sum(indi == graph_labels)
                correct_label += correct.item()

                # print(f"Protein name is : {protein_id}, and its activate is {int(indi.cpu())}.")
                
        elapsed_time = time.time() - begin_time
        print("epoch {:.4f} with epoch time {:.4f} s".format(epoch, elapsed_time))
        result = correct_label / len(validation_dataloader.dataset)

        # 记录预测的label数据文件
        storage_label_file = prog_args.save_dir + "/" + prog_args.dataset + f"/validation_pred_labels_epoch_{epoch}_top{top_i}.csv"

        with open(storage_label_file, 'w') as f:
            f.write("ID_Card,Real_Label,Predict_Label\n")
            for id_card, real_label, predict_label in zip(id_card_record_list, real_label_record_list, predict_label_record_list):
                f.write(f"{id_card},{real_label},{predict_label}\n")
        f.close()




