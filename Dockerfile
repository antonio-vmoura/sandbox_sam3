# Imagem oficial da NVIDIA com CUDA 12.1 e Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Define variáveis para não travar a instalação (interação de timezone, etc.)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema e o Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Instala o PIP para Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Faz o link simbólico para que o comando "python" chame o "python3.11"
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Define o diretório de trabalho
WORKDIR /workspace

# Copia os arquivos de configuração primeiro (cache do Docker)
COPY pyproject.toml* /workspace/
COPY sam3 /workspace/sam3

#  Instala as dependências do projeto
# O --extra-index-url garante que o PyTorch venha com suporte a CUDA 12.1
# Versões fixadas = as do ambiente em que o pipeline (sam3_seg/) foi validado
# (determinismo bit-exato e retomada testados com estas versões).
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -e ".[train, notebooks]" \
        numpy==1.26.4 timm==1.0.30 hydra-core==1.3.7 omegaconf==2.3.1 pycocotools==2.0.11 \
        fvcore==0.1.5.post20221221 iopath==0.1.10 huggingface_hub==1.32.0 scipy==1.17.1 \
        opencv-python==4.11.0.86 pillow==12.3.0 einops==0.8.2 submitit==1.5.4 tensorboard==2.21.0 && \
    pip install optuna==5.0.0 pandas==3.0.6 pyyaml==6.0.3

# Copia o restante do código
COPY . /workspace

# Pipeline de 5 fases (ver run_pipeline_sam3.sh para os volumes necessários)
CMD ["bash", "/workspace/run_pipeline_sam3.sh"]
