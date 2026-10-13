ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    TZ=Asia/Shanghai

ENV CONDA_DIR=/opt/conda \
    CONDA_ENV=GTmining_env \
    SOFTWARE_ROOT=/opt/software \
    PROJECT_ROOT=/workspace/GTmining


RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git unzip tar ca-certificates \
        build-essential cmake \
        gcc-9 g++-9 \
        libeigen3-dev libgmp-dev libgmpxx4ldbl libmpfr-dev \
        libboost-dev libboost-thread-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/*


RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniconda.sh \
    && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/bin:$PATH


RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda create -y -n ${CONDA_ENV} python=3.10 \
    && conda clean -afy


ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:$PATH


RUN conda install -y -n ${CONDA_ENV} \
        pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 \
        -c pytorch -c nvidia \
    && conda install -y -n ${CONDA_ENV} -c dglteam/label/th21_cu118 dgl \
    && conda clean -afy


ARG PIP_INDEX_URL=https://pypi.org/simple


RUN pip install --no-cache-dir mkl==2024.0.0 numpy==1.26.4 \
    && pip install --no-cache-dir pandas==2.1.4 livelossplot==0.5.5 scikit-learn==1.3.2 \
    && pip install --no-cache-dir openpyxl numba biopython \
    && pip install --no-cache-dir ipython tqdm


WORKDIR ${SOFTWARE_ROOT}

# APBS 3.4.1
RUN wget -q https://github.com/Electrostatics/apbs/releases/download/v3.4.1/APBS-3.4.1.Linux.zip \
    && unzip -q APBS-3.4.1.Linux.zip \
    && rm -f APBS-3.4.1.Linux.zip

# pdb2pqr 2.1.1
RUN wget -q https://github.com/Electrostatics/pdb2pqr/releases/download/v2.1.1/pdb2pqr-linux-bin64-2.1.1.tar.gz \
    && tar -xf pdb2pqr-linux-bin64-2.1.1.tar.gz \
    && rm -f pdb2pqr-linux-bin64-2.1.1.tar.gz

# reduce
RUN git clone https://github.com/rlabduke/reduce.git /tmp/reduce \
    && mkdir -p /tmp/reduce/build/reduce \
    && cd /tmp/reduce/build/reduce \
    && cmake -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/reduce_install ../.. \
    && make -j"$(nproc)" \
    && make install \
    && ([ -f ${SOFTWARE_ROOT}/reduce_install/reduce_wwPDB_het_dict.txt ] \
        || cp /tmp/reduce/reduce_wwPDB_het_dict.txt ${SOFTWARE_ROOT}/reduce_install/) \
    && rm -rf /tmp/reduce

# pymesh
RUN export CC=gcc-9 CXX=g++-9 \
    && git clone https://github.com/PyMesh/PyMesh.git /tmp/PyMesh \
    && cd /tmp/PyMesh \
    && git submodule update --init --recursive \
    && pip install --no-cache-dir -r ./python/requirements.txt \
    && ./setup.py build \
    && ./setup.py install \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j"$(nproc)" \
    && pip install --no-cache-dir numpy==1.23.5 \
    && rm -rf /tmp/PyMesh


# msms
COPY msms.tar.gz /tmp/msms.tar.gz
RUN mkdir -p ${SOFTWARE_ROOT}/msms \
    && tar -xf /tmp/msms.tar.gz -C ${SOFTWARE_ROOT}/msms \
    && rm -f /tmp/msms.tar.gz \
    && mv msms.x86_64Linux2*[0-9] msms
    && test -x ${SOFTWARE_ROOT}/msms/msms \
    && test -x ${SOFTWARE_ROOT}/msms/pdb_to_xyzrn \
    && sed -i "s|\./atmtypenumbers|${SOFTWARE_ROOT}/msms/atmtypenumbers|" \
              ${SOFTWARE_ROOT}/msms/pdb_to_xyzrn \
              ${SOFTWARE_ROOT}/msms/pdb_to_xyzr \
    && grep -q "${SOFTWARE_ROOT}/msms/atmtypenumbers" ${SOFTWARE_ROOT}/msms/pdb_to_xyzrn \
    && grep -q "${SOFTWARE_ROOT}/msms/atmtypenumbers" ${SOFTWARE_ROOT}/msms/pdb_to_xyzr


ENV APBS_BIN=${SOFTWARE_ROOT}/APBS-3.4.1.Linux/bin/apbs \
    MULTIVALUE_BIN=${SOFTWARE_ROOT}/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue \
    PDB2PQR_BIN=${SOFTWARE_ROOT}/pdb2pqr-linux-bin64-2.1.1/pdb2pqr \
    REDUCE_HET_DICT=${SOFTWARE_ROOT}/reduce_install/reduce_wwPDB_het_dict.txt \
    MSMS_BIN=${SOFTWARE_ROOT}/msms/msms \
    PDB2XYZRN=${SOFTWARE_ROOT}/msms/pdb_to_xyzrn \
    PATH=${SOFTWARE_ROOT}/reduce_install/bin:${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:$PATH


WORKDIR ${PROJECT_ROOT}
COPY . ${PROJECT_ROOT}/


RUN mkdir -p ${PROJECT_ROOT}/data


WORKDIR ${PROJECT_ROOT}/diffpool

CMD ["/bin/bash"]