# INF2070 - Reinforcement Learning - Spiral Analisys

> Artigo [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning
via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2506.24119)

# Configuração do Ambiente

## Python

O repositório utiliza a versão 3.10 do Python. No momento atual, a minor version mais recente é 3.10.19

> <https://www.python.org/downloads/>

## Instalação do Conda via ASDF

> [ASDF Getting Stated](https://asdf-vm.com/pt-br/guide/getting-started.html)

Após a instalação do ASDF, execute os comandos

```bash
> asdf install python miniforge3-25.3.1-0
> asdf set python miniforge3-25.3.1-0
```

> Verifique se a versão do Python é a esperada

```bash
> python -V
Python 3.10.19
```

## Sistema Linux com placa NVDIA

#### Instalação do CUDA

1. Checar ambiente de hardware e kernel

```bash
> lspci | grep -i nvidia
uname -r 
01:00.0 VGA compatible controller: NVIDIA Corporation TU117M [GeForce GTX 1650 Mobile / Max-Q] (rev a1)
01:00.1 Audio device: NVIDIA Corporation Device 10fa (rev a1)
6.17.4-200.fc42.x86_64
```

2. Atualizar sistema e kernel

```bash
> sudo dnf update -y
> sudo dnf install -y gcc make kernel-headers kernel-devel dkms
``` 

3. Habilitar RPM Fusion para drivers NVDIA


```bash
> sudo dnf install -y \
  https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \ 
  https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm  
```

4. Instalar Driver NVIDIA

> <https://fedoraproject.org/wiki/Cuda>

```bash
> # after reboot:
nvidia-smi
```

5. Instala NVDIA CUDA Toolkit
> [Instruções de instalação do CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#fedora)
>[Cuda Downloads](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Fedora&target_version=42&target_type=rpm_network)


## Configuração do virtual env

```bash
python -m venv venv
```

## Ativar ambiente virtual

```bash
source venv/bin/activate
``` 

## Instalação de dependências

```
> pip install vllm==0.8.4 && pip install oat-llm==0.2.1
``` 

### Troubleshoots

# CUDA Version

Cause: your system CUDA (nvcc) is 13.0 while PyTorch is built for CUDA 12.4 (cu124). Native C/C++ Python extensions (flash-attn, deepspeed) require the toolkit version to match PyTorch's CUDA. Fix by either installing CUDA 12.4 or using a PyTorch build that targets CUDA 13.0.

Options and commands (Fedora):

1) Inspect current state
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

2) Install CUDA 12.4 (preferred if you keep current PyTorch)
```bash
fedora_ver=$(rpm -E %fedora)
sudo dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/fedora${fedora_ver}/x86_64/cuda-fedora${fedora_ver}.repo"
sudo dnf clean expire-cache
sudo dnf -y install cuda-12-4
# make a stable symlink and env vars
sudo ln -sfn /usr/local/cuda-12.4 /usr/local/cuda
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version   # should show 12.4
```

3) Or upgrade/downgrade PyTorch to match system CUDA (use conda for easiest matching)
- To get a PyTorch build for CUDA 13.0 (if available) with conda:
```bash
conda create -n torch-cuda13 python=3.10 -y
conda activate torch-cuda13
conda install pytorch pytorch-cuda=13.0 -c pytorch -c nvidia -y
```
- Or remove system CUDA and install toolkit matching torch (see option 2).

4) After versions match
- Recreate/activate your virtualenv, set CUDA_HOME, then reinstall the failing packages:
```bash
pip install --upgrade pip wheel setuptools
pip install --use-pep517 flash-attn
pip install deepspeed==0.16.8
```

Notes:
- Secure Boot can block NVIDIA kernel modules; disable or sign modules if drivers fail to load.
- If you prefer to avoid compiling extensions, use CPU-only builds or conda wheels built for your CUDA version.
- Verify with python snippet above before reinstalling extensions.

If you want, tell me whether you prefer (A) install CUDA 12.4 on the system or (B) switch PyTorch to CUDA 13 via conda and I’ll give the exact step-by-step commands for your Fedora version.