# GTmining

## Installation dependency(安装依赖)

Successfully installed and tested on Ubuntu 22.04.5 LTS. The C and CXX compilers used were both GNU 9.5.0.

已在Ubuntu 22.04.5 LTS上成功安装并测试。使用的C和CXX编译器都是GNU 9.5.0。

```
# Install system dependency libraries (安装系统依赖库)
sudo apt-get install \
libeigen3-dev libgmp-dev libgmpxx4ldbl libmpfr-dev \
libboost-dev libboost-thread-dev libtbb-dev
```

```
# Create a conda environment
conda create -n GTmining_env python=3.10
conda activate GTmining_env

# Install pytorch, torchvision, torchaudio and pytorch-cuda 
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mkl==2024.0.0
pip install numpy==1.26.4

# Install dgl 
conda install -c dglteam/label/th21_cu118 dgl

# Install other dependencies
pip install pandas==2.1.4
pip install livelossplot==0.5.5
pip install scikit-learn==1.3.2
pip install openpyxl
pip install numba
pip install biopython
```

## MaSIF related dependence (MaSIF相关依赖)

The independently installed software in this tutorial is installed under/home/username/software/, and can be configured in authorized places according to the situation during actual use.

本次教程中独立安装的软件安装在/home/username/software/下面，实际使用时可以根据情况自行配置在有权限的地方。

### Download APBS from following link, unzip it, and then set the environment variable APBS_BIN and MULTIVALUE_BIN.

```
cd /home/username/software/
wget https://github.com/Electrostatics/apbs/releases/download/v3.4.1/APBS-3.4.1.Linux.zip
unzip APBS-3.4.1.Linux.zip
export APBS_BIN=/home/username/software/APBS-3.4.1.Linux/bin/apbs
export MULTIVALUE_BIN=/home/username/software/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue
```

### Download pdb2pqr from following link, unzip it, and then set the environment variable PDB2PQR_BIN.

```
cd /home/username/software/
wget https://github.com/Electrostatics/pdb2pqr/releases/download/v2.1.1/pdb2pqr-linux-bin64-2.1.1.tar.gz
tar -xvf pdb2pqr-linux-bin64-2.1.1.tar.gz
export PDB2PQR_BIN=/home/username/software/pdb2pqr-linux-bin64-2.1.1/pdb2pqr
```

### Download reduce from following link, build it, and then set the environment variable PATH and REDUCE_HET_DICT.

```
cd /home/username/software/
mkdir reduce_install
git clone https://github.com/rlabduke/reduce
cd reduce
mkdir -p ./build/reduce
cd ./build/reduce
cmake -DCMAKE_INSTALL_PREFIX=/home/username/software/reduce_install ../../
make
sudo make install
export PATH=$PATH:/home/username/software/reduce_install/bin
export REDUCE_HET_DICT=/usr/loacl/reduce_wwPDB_het_dict.txt
```

### Download pymesh from following link, and then build it.

<details>
<summary>reference prorocal</summary>

https://www.cnblogs.com/crpfs/p/16180307.html#2-%E4%B8%8B%E8%BD%BD%E7%BC%96%E8%AF%91%E5%B9%B6%E5%AE%89%E8%A3%85-pymesh-%E5%BA%93
https://github.com/PyMesh/PyMesh

</details>

```
cd /home/username/software/
git clone https://github.com/PyMesh/PyMesh.git

cd PyMesh
# it will take several minutes
git submodule update --init --recursive 

pip install -r ./python/requirements.txt

./setup.py build
./setup.py install
mkdir build
cd build
cmake ..
make -j 10
pip install numpy==1.23.5
```

### Download msms from following link, unzip it, and then set the environment variable MSMS_BIN and PDB2XYZRN.

```
cd /home/username/software/

wget -O msms_i86_64Linux2_2.6.1.tar.gz "https://ccsb.scripps.edu/msms/download/933/?tmstv=1783466363"
mkdir msms
tar -xvf msms_i86_64Linux2_2.6.1.tar.gz -C msms
export MSMS_BIN=/home/username/software/msms/msms.x86_64Linux2.2.6.1
export PDB2XYZRN=/home/username/software/msms/pdb_to_xyzrn
```

### Make sure to set following environment variables in your ~/.bashrc file, and then run source ~/.bashrc to make the changes take effect.

```
export APBS_BIN=/home/username/software/APBS-3.4.1.Linux/bin/apbs
export MULTIVALUE_BIN=/home/username/software/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue
export PDB2PQR_BIN=/home/username/software/pdb2pqr-linux-bin64-2.1.1/pdb2pqr
export PATH=$PATH:/home/username/software/reduce_install/bin
export REDUCE_HET_DICT=/home/username/software/reduce_install/reduce_wwPDB_het_dict.txt
export MSMS_BIN=/home/username/software/msms/msms.x86_64Linux2.2.6.1
export PDB2XYZRN=/home/username/software/msms/pdb_to_xyzrn
```

## Download data (下载数据)

Data upload to https://zenodo.org/records/20592146 , download and place in ./data/ directory, and decompress it with the following code:

数据上传至https://zenodo.org/records/20592146，下载后放置在./data/目录下，并使用下述代码解压：

```
cd /home/username/work/
git clone https://github.com/toolManAYY/GTmining.git
cd GTmining
wget https://zenodo.org/records/20592146/files/dl_data.tar.xz?download=1 -O dl_data.tar.xz
mkdir -p ./data/
tar -Jxvf dl_data.tar.xz -C ./data/
```

If the above steps are completed correctly, the directory structure of ./GTmining/ should be as follows.

如果正确完成上述步骤，./GTmining/的目录结构应该如下。

```
├── data
│   └── dl_data
│       ├── GTA_alldata_id
│       └── GTB_alldata_id
└── diffpool
    ├── exe
    ├── MaSIF
    ├── model
    │   ├── dgl_layers
    │   └── tensorized_layers
    ├── model_param_alldata
    └── NGTLYQ

```


## Using tutorials (使用教程)

- The structure file should be placed in a subfolder under diffpool, with the name format: protein_name.pdb.
- Try to ensure that the file name does not contain special characters. At present, the compatible characters tested include letters, numbers, underscores and dots.
- For example, the structure is placed in the ./diffpool/NGTLYQ/ folder, the structure name is CM127523.1_61_BtHGT.pdb

Note: the program currently does not support the prediction of one structure. Please ensure that there are at least two structures in the folder.

- 结构文件应放置在diffpool下的子文件夹中，命名格式为：protein_name.pdb.
- 尽量保证文件名中不包含特殊字符，目前测试兼容的字符有：字母、数字、下划线和点。
- 例如：结构放置在./diffpool/NGTLYQ/文件夹中，结构名称为CM127523.1_61_BtNGT.pdb

Note: 程序目前暂不支持一个结构的预测，请确保文件夹中至少有2个结构。

```
cd diffpool
python predict_donor_specificity.py --input_path ./NGTLYQ/ --type GTB --output_prefix NGT_results
```




