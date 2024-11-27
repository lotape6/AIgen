# AIgen
A test application for LLM AI agent implementation using RAG


# Installation

## Python -> Python3.12

```
sudo apt install python3.12 python3.12-venv
```

## CUDA -> CUDA 12.5

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

## Drivers -> Nvidia-driver-555

```
sudo add-apt-repository ppa:graphics-drivers/ppa --yes 
sudo apt update

sudo apt-get install -y nvidia-driver-555
sudo apt-get install -y cuda-drivers-555

# Reboot
```

## AIgen

```
python3.12 -m venv .venv
source .venv/bin/activate

CMAKE_ARGS="-DGGML_CUDA=on"  pip install "llama-cpp-python<0.3.0" --upgrade --force-reinstall --no-cache-dir

# Install the rest of dependencies
pip3 install -r requirements.txt
```

Outdated tmp

```
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90-cu124/llama_cpp_python-0.2.90-cp312-cp312-linux_x86_64.whl 

python -m pip install llama-index-llms-llama-cpp "llama_cpp_python >= 0.3.2"

# Check you have CUDA >= 12.2 && <= 12.5 and configure the
CMAKE_ARGS="-DLLAMA_CURL=on -DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.6 -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.6/lib64" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir


```