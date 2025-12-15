# GTmining

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
```






