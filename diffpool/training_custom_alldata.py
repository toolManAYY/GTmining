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
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
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

print("{:=^100}".format(f'fold num is : {fold_num}, family type is : {family_fold_type}'))

print("{:=^100}".format('prog_args'))
prog_args = argparse.Namespace(dataset=f'GTmining_6_6_{family_fold_type}_fold{fold_num}', pool_ratio=0.10, num_pool=1, cuda=1, lr=1.0, clip=float("inf"),
                               batch_size=128, epoch=1000, n_worker=10, gc_per_block=3, aggregator_type="meanpool",
                               dropout=0.00, method="diffpool", bn=True, bias=True, save_dir=f"./model_param_alldata",
                               load_epoch=-1, data_mode="default", linkpred=False, hidden_dim=64, embedding_dim=64, family_fold_type=family_fold_type)
print( textwrap.fill(str(prog_args), width=100))

print("{:=^100}".format('加载数据'))
dataset_train = tu.RhaFinderDataset(name="GTmining",
                                    raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/train/')
dataset_validation = tu.RhaFinderDataset(name="GTmining",
                                   raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/validation/')
dataset_test = tu.RhaFinderDataset(name="GTmining",
                                   raw_dir=f'../data/dl_data/{family_fold_type}_alldata/fold{fold_num}/test/')
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