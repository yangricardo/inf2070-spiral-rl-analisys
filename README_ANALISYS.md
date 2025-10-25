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

## Configuração do virtual env

```bash
python -m venv venv
```

## Ativar ambiente virtual

```bash
source venv/bin/activate
``` 