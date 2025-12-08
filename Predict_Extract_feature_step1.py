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

# 获取命令行参数
args = sys.argv[1:]
pdb_filename = str(args[0].strip())
sample_redies_udp = int(args[1].strip())
sample_redies_sugar = int(args[2].strip())
collapse_type = str(args[3].strip())

# 构建基础路径和文件夹
pdb_path = f'/home/admin123/work/GTmining/diffpool/predict_data/structure_align/'
temp_path = f'/home/admin123/work/GTmining/diffpool/predict_data/temp/'
storage_path = f'/home/admin123/work/GTmining/diffpool/predict_data/local_feature/'

if not os.path.isdir(temp_path):
    os.makedirs(temp_path, exist_ok=True)
if not os.path.isdir(storage_path):
    os.makedirs(storage_path, exist_ok=True)

# 构建基础文件名
original_file = os.path.join(pdb_path, pdb_filename)
protonate_file = os.path.join(temp_path, pdb_filename)
ply_filename = os.path.join(temp_path, pdb_filename.replace('.pdb', '.ply'))
storage_filename = os.path.join(storage_path, pdb_filename.replace('.pdb', '.npy'))

# Protonated the pdb structure. 
protonate(original_file, protonate_file)

# Compute MSMS of surface w/hydrogens.
vertices1, faces1, normals1, names1, areas1 = computeMSMS(protonate_file, protonate=True)
# Compute "charged" vertices
vertex_hbond = computeCharges(protonate_file, vertices1, names1)
# For each surface residue, assign the hydrophobicity of its amino acid. 
vertex_hphobicity = computeHydrophobicity(names1)

vertices2 = vertices1
faces2 = faces1

# Fix the mesh.
mesh = pymesh.form_mesh(vertices2, faces2)
mesh_original = mesh # ==========测试
regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
mesh_fixed = regular_mesh # ==========测试
# Compute the normals
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)

# Assign charges on new vertices based on charges of old vertices (nearest neighbor)
vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)
vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)

vertex_charges = computeAPBS(regular_mesh.vertices, protonate_file, protonate_file.split('.pdb')[0])

iface = np.zeros(len(regular_mesh.vertices))
# Compute the surface of the entire complex and from that compute the interface.
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
save_ply(ply_filename, regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal, \
        charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, \
            hphob=vertex_hphobicity, iface=iface)

print(f"Finished extract features from the strucutr {pdb_filename.split('.pdb')[0]}. Thanks for your using!")





