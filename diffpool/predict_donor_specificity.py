import argparse
import os
import subprocess
from tqdm import tqdm
import stat

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
import sys

import shutil
import pandas as pd

import textwrap
import time
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import tu
from model.encoder import DiffPool
from livelossplot import PlotLosses
from sklearn.metrics import f1_score
from collections import defaultdict, Counter




# ============================== args input ==============================
def delete_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


def validate_path(value):
    """Verification structure path: cannot be empty, and the path must exist"""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("Structure path cannot be empty")
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(f"Structure path does not exist: {value}")
    return os.path.abspath(value)


def validate_fold_type(value):
    """Verification fold type: can only be GTA or GTB"""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("Fold type cannot be empty")
    value = value.strip().upper()
    if value not in ("GTA", "GTB"):
        raise argparse.ArgumentTypeError(f"Fold type can only be GTA or GTB, current input: {value}")
    return value


def validate_output_prefix(value):
    """Verification output prefix: cannot be empty, no invalid characters,
    and the parent directory (if specified) must exist and be writable."""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("Output prefix cannot be empty")

    value = value.strip()

    # Check if the prefix contains invalid characters (Windows / Linux compatible)
    invalid_chars = '<>:"|?*'
    if any(c in value for c in invalid_chars):
        raise argparse.ArgumentTypeError(f"Output prefix contains invalid characters: {value}")

    # Get the parent directory of the prefix, so we can verify writability
    parent_dir = os.path.dirname(value) or "."
    if not os.path.exists(parent_dir):
        raise argparse.ArgumentTypeError(f"Parent directory does not exist: {parent_dir}")
    if not os.access(parent_dir, os.W_OK):
        raise argparse.ArgumentTypeError(f"Parent directory is not writable: {parent_dir}")

    return os.path.abspath(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Prediction of donor specificity for glycosyltransferases")
    parser.add_argument(
        "-i", "--input_path", required=True, type=validate_path, help="Structure path, must exist (required)", metavar="PATH"
    )
    parser.add_argument(
        "-t", "--type", required=True, type=validate_fold_type, help="Fold type, optional values: GTA / GTB (required)", metavar="TYPE"
    )
    parser.add_argument(
        "-o", "--output_prefix", required=True, type=validate_output_prefix, help="Output prefix for temporary folders and output files (required)", metavar="PREFIX"
    )

    return parser.parse_args()


args = parse_args()
print("="*20 + " Please check the input parameters: " + "="*20)
print(f"{'Structure path':<40s}: {args.input_path}")
print(f"{'Fold type':<40s}: {args.type}")
print(f"{'Output prefix':<40s}: {args.output_prefix}")

superposition_output_path = f"{args.output_prefix}_superposition/"
os.makedirs(superposition_output_path, exist_ok=True)
delete_files_in_folder(superposition_output_path)
print(f"{'Superposition output path':<40s}: {superposition_output_path}")



# ============================== structure superposition ==============================
print("="*20 + " Superposing structures with USalign " + "="*20)
def get_all_files(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def delete_special_files_in_folder(folder_path, prefix, fila_n):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(prefix) and file.startswith(fila_n):
                file_path = os.path.join(root, file)
                os.remove(file_path)


def validate_executable(exe_path: str) -> str:
    """
    check if the executable file exists and has execute permissions.
    """
    # Existence check
    if not os.path.exists(exe_path):
        raise FileNotFoundError(
            f"Executable not found: {exe_path}\n"
            f"Please confirm that the file is placed in the project directory, or check if the path is spelled correctly."
        )

    # Check for files (not directories)
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(
            f"Specified path is a directory, not an executable file: {exe_path}\n"
            f"Please confirm that exe_path points to the correct executable file."
        )

    # Executable permission check
    # Check whether the current process has execution permission on the file.
    if not os.access(exe_path, os.X_OK):
        raise PermissionError(
            f"Executable exists but lacks execute permissions: {exe_path}\n"
            f"Please run the following command in the terminal to grant execute permissions:\n"
            f"chmod +x {exe_path}"
        )

    # Further check that at least one bit in owner/group/other has x bits
    # Os.access may not be accurate enough in some edge situations (such as permissions granted through ACL)，
    # A cross validation with stat will be more rigorous.
    file_mode = os.stat(exe_path).st_mode
    if not (file_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)):
        raise PermissionError(
            f"Executable permissions do not include any execute bits: {exe_path}\n"
            f"Please run the following command in the terminal to grant execute permissions:\n"
            f"chmod +x {exe_path}"
        )

    return os.path.abspath(exe_path)


fold_type = args.type

directory = args.input_path
all_files = get_all_files(directory)
exe_path = './exe/USalign'
try:
    exe_path = validate_executable(exe_path)
except (FileNotFoundError, PermissionError) as e:
    print(f"\n{'='*60}")
    print(f"Usalign initialization failed and the program terminated.")
    print(f"{'='*60}")
    print(str(e))
    raise SystemExit(1)
cluster_center_pdb = f'./cazy_cluster_center_{fold_type}.pdb'
# USalign AAA.pdb cluster.pdb -o AAA
for file in tqdm(all_files, desc="Superposing structures with USalign", unit="file"):
    file = file.split('/')[-1]
    temp1 = os.path.join(directory, file)
    temp2 = os.path.join(superposition_output_path, file.split('.pdb')[0])
    subprocess.run([exe_path, temp1, cluster_center_pdb, '-o', temp2], stdout=subprocess.DEVNULL)
    delete_special_files_in_folder(superposition_output_path, '.pml', file.split('.pdb')[0])



# ============================== extract surface landscape of active architecture ==============================
pdb_path = superposition_output_path
temp_path = f"{args.output_prefix}_temp/"
local_feature_path = f"{args.output_prefix}_local_feature/"
os.makedirs(temp_path, exist_ok=True)
delete_files_in_folder(temp_path)
os.makedirs(local_feature_path, exist_ok=True)
delete_files_in_folder(local_feature_path)
print(f"{'Temporary files path':<40s}: {temp_path}")

print("="*20 + " Extracting surface landscape of active architecture: Step 1 " + "="*20)
print("This will take tens of seconds to process a structure, please be patient.")
for f in tqdm(os.listdir(pdb_path), desc="Extracting surface landscape Step 1", unit="file"):
    if not f.endswith('.pdb'):
        continue
    original_file = os.path.join(pdb_path, f)
    # --- Protonated the pdb structure. ---
    protonate_file = os.path.join(temp_path, f)
    protonate(original_file, protonate_file)

    # --- Compute MSMS of surface w/hydrogens. ---
    vertices1, faces1, normals1, names1, areas1 = computeMSMS(protonate_file, protonate=True)

    # --- Compute "charged" vertices ---
    vertex_hbond = computeCharges(protonate_file, vertices1, names1)

    # --- For each surface residue, assign the hydrophobicity of its amino acid.  ---
    vertex_hphobicity = computeHydrophobicity(names1)

    # --- Fix the mesh. ---
    mesh = pymesh.form_mesh(vertices1, faces1)
    regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

    # --- Compute the normals ---
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

    # --- Assign charges on new vertices based on charges of old vertices (nearest neighbor) ---
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)

    # --- Compute APBS charges ---
    vertex_charges = computeAPBS(regular_mesh.vertices, protonate_file, protonate_file.split('.pdb')[0])

    iface = np.zeros(len(regular_mesh.vertices))
    # --- Compute the surface of the entire complex and from that compute the interface. ---
    v3, f3, _, _, _ = computeMSMS(protonate_file, protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d) # Square d, because this is how it was in the pyflann version.
    assert(len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.
    ply_filename = os.path.join(temp_path, f.replace('.pdb', '.ply'))
    save_ply(ply_filename, regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal, \
            charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, \
                hphob=vertex_hphobicity, iface=iface)


params = masif_opts['ligand']
print("="*20 + " Extracting surface landscape of active architecture: Step 2 " + "="*20)
print("This will take tens of seconds to process a structure, please be patient.")

def clean_mesh(vertices, edges, component_threshold=3):
    """Cleaning outliers and small components"""
    n = len(vertices)

    # Build adjacency table
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Find connected components
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

    # Keep larger components
    valid = set()
    for comp in components:
        if len(comp) >= component_threshold:
            valid.update(comp)

    # Iteratively delete vertices with degree <=1
    current_valid = valid.copy()
    while True:
        degrees = {u: len([v for v in adj[u] if v in current_valid]) for u in current_valid}
        to_remove = {u for u, d in degrees.items() if d <= 1}
        if not to_remove:
            break
        current_valid -= to_remove

    # Generate cleaning mask
    mask_clean = np.zeros(n, dtype=bool)
    mask_clean[list(current_valid)] = True
    return mask_clean

collapse_type = args.type
if collapse_type == 'GTA':
    NDP_points = np.array([[ 1.603, 19.007, 10.355], [-0.716, 19.148, 10.857], [ 4.897, 18.192, 11.585], [ 3.434, 18.433, 11.889],
                        [ 4.877, 17.968, 10.078], [ 2.986, 19.226, 10.672], [ 4.610, 16.520,  9.690], [ 0.598, 19.410, 11.266],
                        [ 1.254, 18.402,  9.154], [-0.003, 18.158,  8.777], [-1.113, 18.547,  9.672], [ 3.793, 18.777,  9.572],
                        [ 5.658, 19.372, 11.844], [ 3.247, 19.141, 13.097], [ 5.595, 16.103,  8.757], [ 7.677, 14.976,  7.985],
                        [ 0.831, 19.954, 12.347], [ 7.989, 12.795,  6.698], [ 8.490, 12.923,  9.278], [ 7.015, 14.821, 10.445],
                        [ 7.877, 17.085,  9.421], [-2.280, 18.338,  9.355], [ 8.487, 13.560,  7.905], [ 7.110, 15.788,  9.285]])
    SUGAR_points = np.array([[12.831, 13.363,  7.049], [13.291, 14.698,  6.461], [11.604, 12.848,  6.296], [12.121, 15.688,  6.408],
                            [10.510, 13.918,  6.253], [12.497, 16.997,  5.719], [11.014, 15.129,  5.683], [ 9.977, 14.128,  7.560],
                            [13.877, 12.400,  6.955], [13.783, 14.492,  5.135], [11.093, 11.677,  6.924], [13.052, 17.872,  6.684]])
elif collapse_type == 'GTB':
    NDP_points = np.array([[ 0.297,-10.026, -4.534], [ 1.190,-11.657, -5.720], [ 2.245, -9.736, -3.099],
                        [ 4.071,-11.142, -3.509], [ 4.264, -9.568, -1.751], [-0.564, -8.645, -2.688],
                        [-0.287, -7.148, -2.837], [-0.679, -9.071, -4.101], [-1.062, -6.875, -4.110],
                        [-0.659, -5.560, -4.737], [ 1.542,-10.285, -4.082], [ 0.099,-10.888, -5.550],
                        [ 2.123,-11.301, -4.806], [ 3.402,-11.734, -4.516], [ 3.494,-10.141, -2.797],
                        [-0.670, -7.937, -4.909], [-1.792, -8.725, -1.997], [-0.746, -6.425, -1.758],
                        [-1.640, -4.598, -4.574], [-1.563, -0.069, -2.051], [-0.582, -2.255, -3.484],
                        [ 0.211, -3.048, -5.970], [-2.242, -2.264, -5.503], [ 0.374,  0.232, -4.051],
                        [-2.079, -0.374, -4.565], [ 3.932,-12.663, -5.180], [-1.092, -3.046, -4.863],
                        [-0.993, -0.598, -3.527]])
    SUGAR_points = np.array([[-4.035,  0.768, -0.403], [-3.632,  0.106,  0.906], [-3.843, -0.224, -1.509],
                            [-2.203, -0.359,  0.735], [-2.511, -0.911, -1.485], [-1.687, -0.971,  2.039],
                            [-2.136, -1.352, -0.224], [-1.563, -0.069, -2.051], [-5.384,  1.128, -0.393],
                            [-3.642,  1.071,  1.926], [-4.888, -1.150, -1.576], [-0.922, -2.091,  1.736]])

for f in tqdm(os.listdir(temp_path), desc="Extracting surface landscape Step 2", unit="file"):
    if not f.endswith('.ply'):
        continue
    output_dict = {} # vertice_number, xyz, neigh_indecies, si, ddc, hbond, charge, hphob, rho, theta

    # Compute shape complementarity between the two proteins.
    rho = {}
    neigh_indices = {}
    mask = {}
    input_feat = {}
    theta = {}
    iface_labels = {}
    verts = {}

    ply_filename = os.path.join(temp_path, f)
    input_feat, rho, theta, mask, neigh_indices, iface_labels, verts, faces = read_data_from_surface(ply_filename, params)

    sample_redies_udp = 6.0
    sample_redies_sugar = 6.0
    distances = np.full((verts.shape[0],), False, dtype=bool)
    for point in NDP_points:
        distances_temp = np.sqrt(np.sum((verts - point) ** 2, axis=1))
        distances_temp = distances_temp < sample_redies_udp
        distances = distances | distances_temp
    for point in SUGAR_points:
        distances_temp = np.sqrt(np.sum((verts - point) ** 2, axis=1))
        distances_temp = distances_temp < sample_redies_sugar
        distances = distances | distances_temp

    # Get local mesh
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    edges = np.array(list(edges))

    true_indices = np.where(distances)[0]
    new_indices = np.arange(len(true_indices))
    index_mapping = {original_index: new_index for original_index, new_index in zip(true_indices, new_indices)}

    local_edge = []
    for e in edges:
        if e[0] in true_indices and e[1] in true_indices:
            local_edge.append([index_mapping[e[0]], index_mapping[e[1]]])

    output_dict['xyz'] = verts[distances, :]
    output_dict['edges'] = np.array(local_edge)
    output_dict['si'] = input_feat[:, :, 0][distances, :]
    output_dict['hbond'] = input_feat[:, :, 2][distances, :]
    output_dict['charge'] = input_feat[:, :, 3][distances, :]
    output_dict['hphob'] = input_feat[:, :, 4][distances, :]

    # clean mesh
    vertices_sampled = output_dict['xyz']
    edges_sampled = output_dict['edges']
    mask_clean = clean_mesh(vertices_sampled, edges_sampled)

    output_dict['xyz'] = output_dict['xyz'][mask_clean]
    output_dict['edges'] = np.array([[u, v] for u, v in edges_sampled if mask_clean[u] and mask_clean[v]])

    true_indices_clean = np.where(mask_clean)[0]
    index_mapping_clean = {old: new for new, old in enumerate(true_indices_clean)}
    output_dict['edges'] = np.array([[index_mapping_clean[u], index_mapping_clean[v]] for u, v in output_dict['edges']])

    for key in ['si', 'hbond', 'charge', 'hphob']:
        output_dict[key] = output_dict[key][mask_clean]
        output_dict[key] = output_dict[key][:,0:1]

    # Save data only if everything went well. 
    storage_filename = os.path.join(local_feature_path, f.replace('.ply', '.npy'))
    np.save(storage_filename, output_dict)



# ============================== generate predict database ==============================
print("="*20 + " Generate Predict Database " + "="*20)
fold_type = args.type

sample_redies_udp = 6.0
sample_redies_sugar = 6.0

if fold_type == 'GTA':
    graph_label_dict = {'UDP-Glc': 0, 'UDP-GlcNAc': 1, 'UDP-GlcA': 2,
                        'UDP-Gal': 3, 'UDP-GalNAc': 4,
                        'UDP-Xyl': 5, 'GDP-Man': 6,
                        'dTDP-Rha': 7, 'Other': 8}
elif fold_type == 'GTB':
    graph_label_dict = {'UDP-Glc': 0, 'UDP-GlcNAc': 1, 'UDP-GlcA': 2,
                        'UDP-Gal': 3, 'UDP-GalNAc': 4,
                        'UDP-Xyl': 5, 'GDP-Man': 6, 'GDP-Fuc': 7,
                        'dTDP-Rha': 8, 'Other': 9}
else:
    raise ValueError('Fold_type must be one of the GTA and GTB.')

local_feature_path = f"{args.output_prefix}_local_feature/"
dl_data_path = f"{args.output_prefix}_dl_data/"

local_list = [x.split('.npy')[0] for x in os.listdir(local_feature_path)]

# Generate file list
predict_process_path = []
for file in os.listdir(local_feature_path):
    npy_path = os.path.join(local_feature_path, file)
    predict_process_path.append(npy_path)

# Manage folders
if not os.path.isdir(dl_data_path):
    os.makedirs(dl_data_path, exist_ok=True)
else:
    shutil.rmtree(dl_data_path)
    os.makedirs(dl_data_path, exist_ok=True)
os.makedirs(f'{dl_data_path}/predict/')

# generate data
def make_database(process_path: list, data_type: str):
    f_A = open(f'{dl_data_path}/{data_type}/GTmining_A.txt', 'w')
    f_graph_indicator = open(f'{dl_data_path}/{data_type}/GTmining_graph_indicator.txt', 'w')
    f_graph_labels = open(f'{dl_data_path}/{data_type}/GTmining_graph_labels.txt', 'w')
    f_node_attributes = open(f'{dl_data_path}/{data_type}/GTmining_node_attributes.txt', 'w')
    f_itemID = open(f'{dl_data_path}/{data_type}/GTmining_itemID.txt', 'w')
    edge_max = -1
    graph_idx = -1
    for file in tqdm(process_path):
        try:
            input_dict = np.load(file, allow_pickle=True)
        except:
            print(f"wrong {file}")
            continue
        input_dict = input_dict[()]
        if len(input_dict['edges']) <= 100:
            # Used to check for incorrect local meshes
            print(f"error local feature in {file.split('/')[-1].split('.npy')[0]}")
            continue
        item_id = file.split('/')[-1].split('.npy')[0]
        # ++ f_A ++
        edges = input_dict['edges']
        edges += (edge_max +1)
        for edge in edges:
            f_A.write(f"{edge[0]}, {edge[1]}\n")
        edge_max = np.max(edges)
        # ++ f_graph_indicator ++
        graph_idx += 1
        xyzs = input_dict['xyz']
        for i in range(0, xyzs.shape[0]):
            f_graph_indicator.write(f"{graph_idx}\n")
        # ++ f_graph_labels++
        f_graph_labels.write("0\n")
        # ++ f_itemID ++
        f_itemID.write(f"{item_id}\n")
        # ++ f_node_attributes++
        xyzs = input_dict['xyz']
        sis = input_dict['si']
        hbonds = input_dict['hbond']
        charges = input_dict['charge']
        hphobs = input_dict['hphob']
        for i in range(0, xyzs.shape[0]):
            x = xyzs[i][0]
            y = xyzs[i][1]
            z = xyzs[i][2]
            f_node_attributes.write("{:>10.6f}, {:>10.6f}, {:>10.6f}, {:>10.6f}, {:>10.6f}, {:>10.6f}, {:>10.6f}\n".format(x, y, z, sis[i][0], hbonds[i][0], charges[i][0], hphobs[i][0]))

    f_A.close()
    f_graph_indicator.close()
    f_graph_labels.close()
    f_itemID.close()
    f_node_attributes.close()

make_database(predict_process_path, 'predict')



# ============================== predict donor specificity ==============================
print("="*20 + " Predict Donor Specificity " + "="*20)

fold_num = 1
family_fold_type = args.type

def prepare_data(dataset, shuffle=False, prog_args=None, custom_batch_size=None):
    """
    preprocess TU dataset according to DiffPool's paper setting and load dataset into dataloader
    """
    if custom_batch_size is None:
        return dgl.dataloading.GraphDataLoader(
            dataset,
            batch_size=prog_args.batch_size,
            shuffle=shuffle,
            num_workers=prog_args.n_worker,
        )
    else:
        return dgl.dataloading.GraphDataLoader(
            dataset,
            batch_size=custom_batch_size,
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
        self.item_ids = []
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
            skk = 1
            #  print(f"Warning: Indices in file seem to be 0-based (min={min_val}). Proceeding assuming 0-based.")
            return idx_tensor
        else:
            # Standard 1-based to 0-based conversion
            return idx_tensor - 1 # More standard than subtracting min if known to start at 1

    def process(self):
        """
        Loads data from text files and constructs a list of DGLGraphs.
        """
        # print(f"Processing custom dataset: {self.name}")

        # --- 1. Load Edge List ---
        # print(f"Loading edges from {self._file_path('A')}")
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
        # print(f"Loading graph indicators from {self._file_path('graph_indicator')}")
        node_graph_ids_raw = np.loadtxt(self._file_path("graph_indicator"), dtype=int)
        # Convert graph IDs to 0-based indices
        node_graph_ids = self._idx_from_zero(node_graph_ids_raw)
        num_total_nodes_in_file = len(node_graph_ids_raw)

        # --- 3. Load Graph Labels (might be dummy) ---
        # print(f"Loading graph labels from {self._file_path('graph_labels')}")
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

        # --- 4. Load Item IDs ---
        # print(f"Loading item IDs from {self._file_path('itemID')}")
        try:
            # Read itemid file, support integer or string type ID
            self.item_ids = np.loadtxt(self._file_path("itemID"), dtype=str).tolist()
            
            # Verify that the number of item IDs matches the number of graphs
            num_graphs_in_file = len(set(node_graph_ids))
            if len(self.item_ids) != num_graphs_in_file:
                raise ValueError(
                    f"Number of item IDs ({len(self.item_ids)}) does not match number of graphs ({num_graphs_in_file})."
                )
        except FileNotFoundError:
            print(f"Warning: Item ID file {self._file_path('itemID')} not found. Using graph index as item ID.")
            num_graphs_in_file = len(set(node_graph_ids))
            self.item_ids = [str(i) for i in range(num_graphs_in_file)]  # Using index as default ID


        # --- 4. Load Node Attributes ---
        # print(f"Loading node attributes from {self._file_path('node_attributes')}")
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
        # print(f"Found {num_expected_graphs} graphs based on graph_indicator.")

        for graph_id in range(num_expected_graphs):
            # Find the nodes belonging to the current graph (graph_id)
            node_mask = (node_graph_ids == graph_id)
            node_indices_for_graph = np.where(node_mask)[0] # Get 0-based indices of nodes in this graph

            if len(node_indices_for_graph) == 0:
                print(f"Warning: Graph ID {graph_id} has no nodes according to graph_indicator.")
                g_sub = dgl.graph(([], []), num_nodes=0)
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
        # print(f"Max number of nodes in a single graph: {self.max_num_node}")


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
        item_id = self.item_ids[idx]

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
        # 1. Node feature dimension (return 0 if there is no feature)
        if len(self.graph_lists) > 0 and 'feat' in self.graph_lists[0].ndata:
            input_dim = self.graph_lists[0].ndata['feat'].shape[1]
        else:
            input_dim = 0

        # 2. Label dimension (number of classes)
        label_dim = self.num_classes

        # 3. Maximum number of nodes
        max_num_node = self.max_num_node

        return input_dim, label_dim, max_num_node

print("{:=^100}".format(f'fold num is : {fold_num}, family type is : {family_fold_type}'))

print("{:=^100}".format('prog_args'))
prog_args = argparse.Namespace(dataset=f'GTmining_6_6_{family_fold_type}_fold{fold_num}', pool_ratio=0.10, num_pool=1, cuda=1, lr=1.0, clip=float("inf"),
                               batch_size=128, epoch=500, n_worker=10, gc_per_block=3, aggregator_type="meanpool",
                               dropout=0.00, method="diffpool", bn=True, bias=True, save_dir=f"./model_param_alldata/",
                               load_epoch=-1, data_mode="default", linkpred=False, hidden_dim=64, embedding_dim=64, family_fold_type=family_fold_type)
print( textwrap.fill(str(prog_args), width=100))

print("{:=^100}".format('Loading Data'))
dataset_train = customreaddata(name="GTmining",
                                    raw_dir=f'../data/dl_data/{family_fold_type}_alldata_id/fold{fold_num}/train/')
dataset_validation = customreaddata(name="GTmining",
                                   raw_dir=f'../data/dl_data/{family_fold_type}_alldata_id/fold{fold_num}/validation/')
dataset_test = customreaddata(name="GTmining",
                                   raw_dir=f'../data/dl_data/{family_fold_type}_alldata_id/fold{fold_num}/test/')
train_dataloader = prepare_data(dataset_train, shuffle=True, prog_args=prog_args)
validation_dataloader = prepare_data(dataset_validation, shuffle=False, prog_args=prog_args)
test_dataloader = prepare_data(dataset_test, shuffle=False, prog_args=prog_args)

dataset_predict = customreaddata(name="GTmining",
                                   raw_dir=f'{args.output_prefix}_dl_data/predict/')
predict_dataloader = prepare_data(dataset_predict, shuffle=False, prog_args=prog_args, custom_batch_size=1)


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

custom_loss_weight = [1.0] * label_dim_train

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

# First pass through train_dataloader to initialize some parameters in the model
model.train()
for batch_idx, (batch_graph, graph_labels, items_id) in enumerate(train_dataloader):
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



# ============================== Start predict donor specificity ==============================
predict_result = defaultdict(list)
predict_result_freq = defaultdict(list)

for fold in range(1, 11):
    result_index = 1

    begin_time = time.time()
    print("\nEPOCH ###### Fold {}######".format(fold))
    if prog_args.save_dir is not None:
        model.load_state_dict(
            torch.load(
                prog_args.save_dir
                + f"/model-GTmining_6_6_{args.type}_fold{fold}" , weights_only=True
            )
        )


    model.eval()
    with torch.no_grad():
        val_pred_indi = torch.tensor([], device='cuda')
        val_label_indi = torch.tensor([], device='cuda')
        for batch_idx, (batch_graph, graph_labels, item_id) in enumerate(predict_dataloader):
            for key, value in batch_graph.ndata.items():
                batch_graph.ndata[key] = value.float()
            graph_labels = graph_labels.long()
            if torch.cuda.is_available():
                batch_graph = batch_graph.to(torch.cuda.current_device())
                graph_labels = graph_labels.cuda()

            # get protein id
            protein_id = item_id[0]


            ypred = model(batch_graph)
            indi = torch.argmax(ypred, dim=1)
            predict_result[protein_id+f'_{result_index}'].append(int(indi.cpu()))
            predict_result_freq[protein_id+f'_{result_index}'].append(F.softmax(ypred, dim=1).cpu().tolist()[0])
            result_index += 1
            
    elapsed_time = time.time() - begin_time
    print("fold {:.4f} with epoch time {:.4f} s".format(fold, elapsed_time))

    # break


temp_predict_result_freq = defaultdict(list)

for key in predict_result_freq.keys():
    freq_list = np.array(predict_result_freq[key])
    freq_list = freq_list.reshape(-1, len(graph_label_dict)).round(2)
    temp_predict_result_freq[key] = freq_list.tolist()

predict_result_freq = temp_predict_result_freq



# ============================== Save Results ==============================
result = {}

for key, values in predict_result.items():
    count = Counter(values)
    
    if not count:
        result[key] = []
        continue

    max_freq = max(count.values())
    most_common = [(num, freq) for num, freq in count.items() if freq == max_freq]
    result[key] = most_common

# print results
for key, items in result.items():
    print(f"{key}:")
    print(predict_result[key])
    print("Predicted probability distribution")
    for i in range(len(predict_result_freq[key])):
        print(predict_result_freq[key][i])
    for num, freq in items:
        print(f"  Element {num}, frequency {freq}")
    print()  # empty line for separation


# Store results as excel table

# reverse graph_label_dict
label_to_name = {v: k for k, v in graph_label_dict.items()}

data = []
for key, items in result.items():
    predictions = predict_result[key]
    for num, freq in items:
        label_name = label_to_name.get(num, "Unknown")
        data.append([key, predictions, f"Element {label_name}, frequency {freq}"])

if args.type == 'GTA':
    graph_label_dict = {'UDP-Glc': 0, 'UDP-GlcNAc': 1, 'UDP-GlcA': 2,
                        'UDP-Gal': 3, 'UDP-GalNAc': 4,
                        'UDP-Xyl': 5, 'GDP-Man': 6,
                        'dTDP-Rha': 7, 'Other': 8}
    print(graph_label_dict)
elif args.type == 'GTB':
    graph_label_dict = {'UDP-Glc': 0, 'UDP-GlcNAc': 1, 'UDP-GlcA': 2,
                        'UDP-Gal': 3, 'UDP-GalNAc': 4,
                        'UDP-Xyl': 5, 'GDP-Man': 6, 'GDP-Fuc': 7,
                        'dTDP-Rha': 8, 'Other': 9}
    print(graph_label_dict)


# 创建DataFrame
df_result = pd.DataFrame(data, columns=["Key", "Predictions", "Most Common Element"])

# 保存为Excel文件
df_result.to_excel(f"{args.output_prefix}_prediction_results.xlsx", index=False)
print(f"Results saved as {args.output_prefix}_prediction_results.xlsx")





