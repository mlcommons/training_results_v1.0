mkdir -p $HOME/dlrm

# install anaconda
cd $HOME/dlrm
wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
chmod +x anaconda3.sh
./anaconda3.sh -b -p ~/anaconda3
~/anaconda3/bin/conda create -n dlrm python=3.7

export PATH=~/anaconda3/bin:$PATH
source ~/anaconda3/bin/activate dlrm

# install depedency packages
pip install sklearn onnx tqdm lark-parser
pip install -e git+https://github.com/mlperf/logging@1.0.0-rc4#egg=logging

conda config --append channels intel
conda install ninja pyyaml setuptools cmake cffi typing
conda install intel-openmp mkl mkl-include numpy -c intel --no-update-deps
conda install -c conda-forge gperftools

# clone PyTorch
cd $HOME/dlrm
git clone https://github.com/pytorch/pytorch.git
cd pytorch && git checkout tags/v1.5.0-rc3 -b v1.5-rc3
git submodule sync && git submodule update --init --recursive

# clone Intel Extension for PyTorch
cd $HOME/dlrm
git clone https://github.com/intel/intel-extension-for-pytorch.git
cd intel-extension-for-pytorch && git checkout tags/v0.2 -b v0.2
git submodule update --init --recursive

# install PyTorch
cd $HOME/dlrm/pytorch
cp $HOME/dlrm/intel-extension-for-pytorch/torch_patches/0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch .
patch -p1 < 0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch
python setup.py install

# install Intel Extension for PyTorch
cd $HOME/dlrm/intel-extension-for-pytorch
wget https://patch-diff.githubusercontent.com/raw/intel/intel-extension-for-pytorch/pull/161.diff 
git apply 161.diff
python setup.py install

# install torch-ccl
cd $HOME/dlrm
git clone https://github.com/intel/torch-ccl.git
cd torch-ccl && git checkout remotes/origin/ccl_torch1.5 -b ccl_torch1.5
git submodule sync && git submodule update --init --recursive
cd third_party/oneCCL
git apply ../../patches/Fix_the_* && cd ../../   
export PATH=/opt/crtdc/binutils/2.36/bin/:$PATH
export CCL_BF16=avx512bf
export LD_LIBRARY_PATH=GCC-10.2.0/lib:GCC-10.2.0/lib64:${LD_LIBRARY_PATH}
CMAKE_C_COMPILER=GCC-10.2.0/bin/gcc CMAKE_CXX_COMPILER=GCC-10.2.0/bin/g++  python setup.py install

# install dlrm
cd $HOME/dlrm
git clone https://github.com/facebookresearch/dlrm.git
cd dlrm && git checkout mlperf
wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/653f6464c651411e9e694265429e4b87402d434e/torch_patches/models/dlrm_mlperf_v1.0_training.diff
git apply dlrm_mlperf_v1.0_training.diff


