# GTmining

:face_with_head_bandage: Detailed inference scripts and model checkpoints are being organized and will be released within one to two weeks.
:face_with_head_bandage: 详细的推理脚本和模型checkpoints正在整理中，将会在1~2周内发布，敬请期待
2026-05-19

## 安装依赖教程
```
conda create -n GTmining_env python=3.10
conda activate GTmining_env

# pytorch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mkl==2024.0.0
pip install numpy==1.26.4

# dgl
conda install -c dglteam/label/th21_cu118 dgl

# 其他依赖
pip install pandas==2.1.4
pip install livelossplot==0.5.5
pip install scikit-learn==1.3.2
pip install openpyxl
pip install numba
pip install biopython

# MaSIF相关依赖
export APBS_BIN=/home/admin123/software/APBS-3.4.1.Linux/bin/apbs
export MULTIVALUE_BIN=/home/admin123/software/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue
export PDB2PQR_BIN=/home/admin123/software/pdb2pqr-linux-bin64-2.1.1/pdb2pqr
export PATH=$PATH:/home/admin123/software/reduce_install/bin
export REDUCE_HET_DICT=/home/admin123/software/reduce_install/reduce_wwPDB_het_dict.txt
export PYMESH_PATH=/path/to/PyMesh
export MSMS_BIN=/home/admin123/software/msms/msms.x86_64Linux2.2.6.1
export PDB2XYZRN=/home/admin123/software/msms/pdb_to_xyzrn

## APBS下载地址=3.4.1版本
https://github.com/Electrostatics/apbs/releases/tag/v3.4.1

## PDB2PQR下载地址=2.1.1版本
https://github.com/Electrostatics/pdb2pqr/releases/tag/v2.1.1

## reduce下载地址
https://github.com/rlabduke/reduce
mkdir -p ./build/reduce
cd ./build/reduce
cmake -DCMAKE_INSTALL_PREFIX=/home/admin123/software/reduce_install ../../
make
make install

## pymesh安装和下载
https://www.cnblogs.com/crpfs/p/16180307.html#2-%E4%B8%8B%E8%BD%BD%E7%BC%96%E8%AF%91%E5%B9%B6%E5%AE%89%E8%A3%85-pymesh-%E5%BA%93
https://github.com/PyMesh/PyMesh
git clone https://github.com/PyMesh/PyMesh.git
### 递归克隆仓库中的第三方源码仓库
cd PyMesh
git submodule update --init --recursive
### 安装 PyMesh 需要的系统依赖库
sudo apt-get install \
libeigen3-dev \
libgmp-dev \
libgmpxx4ldbl \
libmpfr-dev \
libboost-dev \
libboost-thread-dev \
libtbb-dev
### 安装 PyMesh 需要的 Python 依赖库
pip install -r ./python/requirements.txt
### 编译并安装 PyMesh 库
./setup.py build
./setup.py install
mkdir build
cd build
cmake ..
make
pip install numpy==1.23.5

### msms下载地址
https://ccsb.scripps.edu/msms/downloads/

```






