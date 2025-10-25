# INF2070 - Reinforcement Learning - Spiral Analisys

> Artigo [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning
via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/pdf/2506.24119)

# Configuração do Ambiente

## Python

O repositório utiliza a versão 3.10 do Python. No momento atual, a minor version mais recente é 3.10.19

> <https://www.python.org/downloads/>

## Instalação do Python via ASDF

> [ASDF Getting Stated](https://asdf-vm.com/pt-br/guide/getting-started.html)

Após a instalação do ASDF, execute os comandos

```bash
> asdf install python 3.10.19
> asdf set python 3.10.19
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
