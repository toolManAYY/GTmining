from MaSIF.protonate import protonate
from MaSIF.computeMSMS import computeMSMS
from MaSIF.computeCharges import computeCharges, assignChargesToNewMesh
from MaSIF.computeHydrophobicity import computeHydrophobicity
from MaSIF.fixmesh import fix_mesh
from MaSIF.masif_opts import masif_opts
from MaSIF.compute_normal import compute_normal
from MaSIF.computeAPBS import computeAPBS
from MaSIF.save_ply import save_ply
from MaSIF.read_data_from_surface import read_data_from_surface
import numpy as np
from Bio.PDB import *
import pymesh
from sklearn.neighbors import KDTree
import os


import sys


def clean_mesh(vertices, edges, component_threshold=3):
    """清洗孤立点和小组件"""
    n = len(vertices)

    # 构建邻接表
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # 查找连通组件
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            component = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(component)

    # 保留较大组件
    valid = set()
    for comp in components:
        if len(comp) >= component_threshold:
            valid.update(comp)

    # 迭代删除度数<=1的顶点
    current_valid = valid.copy()
    while True:
        degrees = {u: len([v for v in adj[u] if v in current_valid]) for u in current_valid}
        to_remove = {u for u, d in degrees.items() if d <= 1}
        if not to_remove:
            break
        current_valid -= to_remove

    # 生成清洗掩码
    mask_clean = np.zeros(n, dtype=bool)
    mask_clean[list(current_valid)] = True
    return mask_clean


# 获取命令行参数
args = sys.argv[1:]
pdb_filename = str(args[0].strip())
folder_name = str(args[1].strip())
sample_redies_udp = float(args[2].strip())
sample_redies_sugar = float(args[3].strip())


pdb_path = '/home/admin123/work/GTmining/data/cazy_annot_domain/domain/'
temp_path = '/home/admin123/work/GTmining/data/local_features/global_features/'
storage_path = '/home/admin123/work/GTmining/data/local_features/features/'
storage_path = os.path.join(storage_path, f"udp{sample_redies_udp}_sugar{sample_redies_sugar}")
os.makedirs(storage_path, exist_ok=True)
os.makedirs(os.path.join(temp_path, folder_name), exist_ok=True)
os.makedirs(os.path.join(storage_path, folder_name), exist_ok=True)

original_file = os.path.join(pdb_path, folder_name, pdb_filename)
protonate_file = os.path.join(temp_path, folder_name, pdb_filename)
ply_filename = os.path.join(temp_path, folder_name, pdb_filename.replace('.pdb', '.ply'))
storage_filename = os.path.join(storage_path, folder_name, pdb_filename.replace('.pdb', '.npy'))


# ==========Coumpute Data==========

params = masif_opts['ligand']

output_dict = {} # vertice_number, xyz, neigh_indecies, si, ddc, hbond, charge, hphob, rho, theta

# Compute shape complementarity between the two proteins.
rho = {}
neigh_indices = {}
mask = {}
input_feat = {}
theta = {}
iface_labels = {}
verts = {}

input_feat, rho, theta, mask, neigh_indices, iface_labels, verts, faces = read_data_from_surface(ply_filename, params)

UDP_points = np.array([[ 1.603, 19.007, 10.355], [-0.716, 19.148, 10.857], [ 4.897, 18.192, 11.585], [ 3.434, 18.433, 11.889],
                       [ 4.877, 17.968, 10.078], [ 2.986, 19.226, 10.672], [ 4.610, 16.520,  9.690], [ 0.598, 19.410, 11.266],
                       [ 1.254, 18.402,  9.154], [-0.003, 18.158,  8.777], [-1.113, 18.547,  9.672], [ 3.793, 18.777,  9.572],
                       [ 5.658, 19.372, 11.844], [ 3.247, 19.141, 13.097], [ 5.595, 16.103,  8.757], [ 7.677, 14.976,  7.985],
                       [ 0.831, 19.954, 12.347], [ 7.989, 12.795,  6.698], [ 8.490, 12.923,  9.278], [ 7.015, 14.821, 10.445],
                       [ 7.877, 17.085,  9.421], [-2.280, 18.338,  9.355], [ 8.487, 13.560,  7.905], [ 7.110, 15.788,  9.285]])
SUGAR_points = np.array([[12.831, 13.363,  7.049], [13.291, 14.698,  6.461], [11.604, 12.848,  6.296], [12.121, 15.688,  6.408],
                         [10.510, 13.918,  6.253], [12.497, 16.997,  5.719], [11.014, 15.129,  5.683], [ 9.977, 14.128,  7.560],
                         [13.877, 12.400,  6.955], [13.783, 14.492,  5.135], [11.093, 11.677,  6.924], [13.052, 17.872,  6.684]])

distances = np.full((verts.shape[0],), False, dtype=bool)
for point in UDP_points:
   distances_temp = np.sqrt(np.sum((verts - point) ** 2, axis=1))
   distances_temp = distances_temp < sample_redies_udp
   distances = distances | distances_temp
for point in SUGAR_points:
   distances_temp = np.sqrt(np.sum((verts - point) ** 2, axis=1))
   distances_temp = distances_temp < sample_redies_sugar
   distances = distances | distances_temp

# ==================== 获取边的信息 ====================
# 创建一个空的边集来存储边，使用集合来避免重复边
edges = set()
# 遍历每个面，将其顶点连接成边
for face in faces:
    # 获取每个面的三条边，顶点索引两两组合
    edges.add(tuple(sorted([face[0], face[1]])))
    edges.add(tuple(sorted([face[1], face[2]])))
    edges.add(tuple(sorted([face[2], face[0]])))
# 将边转换为numpy数组
edges = np.array(list(edges))

# 使用np.where获取值为True的元素的索引
true_indices = np.where(distances)[0]
# 创建一个从0开始的索引列表，这里的长度与true_indices相同
new_indices = np.arange(len(true_indices))
# 创建映射关系，将原始索引映射到新的索引
index_mapping = {original_index: new_index for original_index, new_index in zip(true_indices, new_indices)}

# 拿到新索引的边
local_edge = []
for e in edges:
    if e[0] in true_indices and e[1] in true_indices:
        local_edge.append([index_mapping[e[0]], index_mapping[e[1]]])

# output_dict['vertice_number'] = int(verts.shape[0])
# output_dict['index_mapping'] = index_mapping
# output_dict['distances'] = distances
output_dict['xyz'] = verts[distances, :]
output_dict['edges'] = np.array(local_edge)
# output_dict['neigh_indecies'] = neigh_indices
output_dict['si'] = input_feat[:, :, 0][distances, :]
# output_dict['ddc'] = input_feat[:, :, 1][distances, :]
output_dict['hbond'] = input_feat[:, :, 2][distances, :]
output_dict['charge'] = input_feat[:, :, 3][distances, :]
output_dict['hphob'] = input_feat[:, :, 4][distances, :]
# output_dict['rho'] = rho[distances, :]
# output_dict['theta'] = theta[distances, :]

# ==================== 网格清洗步骤 ====================
vertices_sampled = output_dict['xyz']
edges_sampled = output_dict['edges']
mask_clean = clean_mesh(vertices_sampled, edges_sampled)

# 更新output_dict中的属性
output_dict['xyz'] = output_dict['xyz'][mask_clean]
output_dict['edges'] = np.array([[u, v] for u, v in edges_sampled if mask_clean[u] and mask_clean[v]])

# 重新映射索引
true_indices_clean = np.where(mask_clean)[0]
index_mapping_clean = {old: new for new, old in enumerate(true_indices_clean)}
output_dict['edges'] = np.array([[index_mapping_clean[u], index_mapping_clean[v]] for u, v in output_dict['edges']])

# 更新其他特征数据
# for key in ['si', 'ddc', 'hbond', 'charge', 'hphob', 'rho', 'theta']:
for key in ['si', 'hbond', 'charge', 'hphob']:
    output_dict[key] = output_dict[key][mask_clean]
    output_dict[key] = output_dict[key][:,0:1]

# 更新index_mapping
# original_to_sampled = output_dict['index_mapping']
# sampled_to_clean = {old: idx for idx, old in enumerate(true_indices_clean)}
# output_dict['index_mapping'] = {orig: sampled_to_clean[sampled] for orig, sampled in original_to_sampled.items() if sampled in sampled_to_clean}


# Save data only if everything went well. 
np.save(storage_filename, output_dict)
print(f"Finished extract features from the strucutr {pdb_filename.rstrip('.pdb')}. Thanks for your using!")





