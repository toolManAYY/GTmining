# GTmining

:face_with_head_bandage: Detailed inference scripts and model checkpoints are being organized and will be released within one to two weeks.

:face_with_head_bandage: 详细的推理脚本和模型checkpoints正在整理中，将会在1~2周内发布，敬请期待

writing in 2026-05-19

## Installation dependency tutorial (安装依赖教程)
```
conda create -n GTmining_env python=3.10
conda activate GTmining_env

# pytorch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mkl==2024.0.0
pip install numpy==1.26.4

# dgl
conda install -c dglteam/label/th21_cu118 dgl

# Other dependencies (其他依赖)
pip install pandas==2.1.4
pip install livelossplot==0.5.5
pip install scikit-learn==1.3.2
pip install openpyxl
pip install numba
pip install biopython

# MaSIF related dependence (MaSIF相关依赖)
export APBS_BIN=/home/admin123/software/APBS-3.4.1.Linux/bin/apbs
export MULTIVALUE_BIN=/home/admin123/software/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue
export PDB2PQR_BIN=/home/admin123/software/pdb2pqr-linux-bin64-2.1.1/pdb2pqr
export PATH=$PATH:/home/admin123/software/reduce_install/bin
export REDUCE_HET_DICT=/home/admin123/software/reduce_install/reduce_wwPDB_het_dict.txt
export PYMESH_PATH=/path/to/PyMesh
export MSMS_BIN=/home/admin123/software/msms/msms.x86_64Linux2.2.6.1
export PDB2XYZRN=/home/admin123/software/msms/pdb_to_xyzrn

## APBS=3.4.1
https://github.com/Electrostatics/apbs/releases/tag/v3.4.1

## PDB2PQR=2.1.1
https://github.com/Electrostatics/pdb2pqr/releases/tag/v2.1.1

## reduce
https://github.com/rlabduke/reduce
mkdir -p ./build/reduce
cd ./build/reduce
cmake -DCMAKE_INSTALL_PREFIX=/home/admin123/software/reduce_install ../../
make
make install

## pymesh
### reference protocol
https://www.cnblogs.com/crpfs/p/16180307.html#2-%E4%B8%8B%E8%BD%BD%E7%BC%96%E8%AF%91%E5%B9%B6%E5%AE%89%E8%A3%85-pymesh-%E5%BA%93
https://github.com/PyMesh/PyMesh
### git clone
git clone https://github.com/PyMesh/PyMesh.git
### Third party source code repository in recursive cloning repository (递归克隆仓库中的第三方源码仓库)
cd PyMesh
git submodule update --init --recursive
### System dependency libraries required for pymesh installation (安装 PyMesh 需要的系统依赖库)
sudo apt-get install \
libeigen3-dev \
libgmp-dev \
libgmpxx4ldbl \
libmpfr-dev \
libboost-dev \
libboost-thread-dev \
libtbb-dev
### Python dependency libraries required for pymesh installation (安装 PyMesh 需要的 Python 依赖库)
pip install -r ./python/requirements.txt
### Compile and install pymesh Libraries (编译并安装 PyMesh 库)
./setup.py build
./setup.py install
mkdir build
cd build
cmake ..
make
pip install numpy==1.23.5

### msms
https://ccsb.scripps.edu/msms/downloads/

```

## Download data (下载数据)

Data upload to https://zenodo.org/records/20592146 , download and place in ./data/ directory, and decompress it with the following code:

数据上传至https://zenodo.org/records/20592146，下载后放置在./data/目录下，并使用下述代码解压：

```
tar -Jxvf dl_data.tar.xz
```


## Using tutorials (使用教程)

- The structure file should be placed in a subfolder under diffpool, with the name format: protein_name.pdb.
- Try to ensure that the file name does not contain special characters. At present, the compatible characters tested include letters, numbers, underscores and dots.
- For example, the structure is placed in the ./diffpool/NGTLYQ/ folder, the structure name is CM127523.1_61_BtHGT.pdb

Note: the program currently does not support the prediction of one structure. Please ensure that there are at least two structures in the folder.

- 结构文件应放置在diffpool下的子文件夹中，命名格式为：protein_name.pdb.
- 尽量保证文件名中不包含特殊字符，目前测试兼容的字符有：字母、数字、下划线和点。
- 例如：结构放置在./diffpool/NGTLYQ/文件夹中，结构名称为CM127523.1_61_BtNGT.pdb

note: 程序目前暂不支持一个结构的预测，请确保文件夹中至少有2个结构。

```
python predict_donor_specificity.py --input_path ./NGTLYQ/ --type GTB --output_prefix NGT_results
```




