# 1. Usamos a imagem oficial da NVIDIA com CUDA 12.1 e Ubuntu 22.04
# A versão "devel" inclui cabeçalhos C++ necessários para compilar extensões (como o SAM)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Define variáveis para não travar a instalação (interação de timezone, etc.)
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Instala dependências do sistema e o Python 3.11
# Como o Ubuntu 22.04 vem com Py3.10, adicionamos o repositório 'deadsnakes' para pegar o 3.11
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

# 3. Instala o PIP para Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

# Faz o link simbólico para que o comando "python" chame o "python3.11"
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Define o diretório de trabalho
WORKDIR /workspace

# 4. Copia os arquivos de configuração primeiro (cache do Docker)
COPY pyproject.toml* /workspace/
COPY sam3 /workspace/sam3

# 5. Instala as dependências do projeto
# O --extra-index-url garante que o PyTorch venha com suporte a CUDA 12.1
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -e ".[train, notebooks]"

# Copia o restante do código
COPY . /workspace

# Comando padrão
CMD ["python", "sam3/train/train.py", "-c", "configs/sam3_ph2_docker.yaml", "--use-cluster", "0"]