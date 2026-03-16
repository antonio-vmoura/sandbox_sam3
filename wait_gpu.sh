#!/bin/bash

# Verifica se o token foi setado para evitar falha após horas de espera
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "ERRO: A variável HUGGING_FACE_HUB_TOKEN não está definida no ambiente."
    exit 1
fi

echo "Aguardando as GPUs 0 e 1 ficarem livres por 5 minutos contínuos..."

CHECK_INTERVAL=60
REQUIRED_IDLE_MINUTES=5
IDLE_COUNT=0

while true; do
    # Captura métricas das GPUs
    GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
    GPU0_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
    GPU1_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1)

    # Condição: Memória < 1000MB E Uso < 10% para AMBAS as GPUs
    if [ "$GPU0_MEM" -lt 1000 ] && [ "$GPU1_MEM" -lt 1000 ] && \
       [ "$GPU0_UTIL" -lt 10 ] && [ "$GPU1_UTIL" -lt 10 ]; then
        
        # Incrementa o contador se estiver ocioso
        ((IDLE_COUNT++))
        echo "$(date) | GPU0: ${GPU0_MEM}MiB ${GPU0_UTIL}% | GPU1: ${GPU1_MEM}MiB ${GPU1_UTIL}% -> Ociosa há $IDLE_COUNT minuto(s)."

        # Verifica se atingiu o tempo necessário
        if [ "$IDLE_COUNT" -ge "$REQUIRED_IDLE_MINUTES" ]; then
            echo "GPUs livres por $REQUIRED_IDLE_MINUTES minutos contínuos! Iniciando treinamento do SAM3..."
            break
        fi
    else
        # Se houve pico de uso, verifica se o contador estava rodando para avisar do reset
        if [ "$IDLE_COUNT" -gt 0 ]; then
            echo "$(date) | Atividade detectada! Resetando contador de ociosidade."
        else
            echo "$(date) | GPU0: ${GPU0_MEM}MiB ${GPU0_UTIL}% | GPU1: ${GPU1_MEM}MiB ${GPU1_UTIL}% -> Em uso."
        fi
        
        # Zera o contador
        IDLE_COUNT=0
    fi

    sleep $CHECK_INTERVAL
done

# Inicia o Docker
# docker run --gpus all -it --rm \
#    --ipc=host \
#    --user $(id -u):$(id -g) \
#    -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
#    -e HF_HOME=/workspace/cache/huggingface \
#    -e TORCH_HOME=/workspace/cache/torch \
#    -e HOME=/workspace/cache \
#    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#    -v $(pwd)/datasets/ph2:/workspace/data \
#    -v $(pwd)/logs:/workspace/logs \
#    -v $(pwd)/sam3:/workspace/sam3 \
#    -v $(pwd)/configs:/workspace/configs \
#    -v $(pwd)/sam3_cache:/workspace/cache \
#    -v /etc/passwd:/etc/passwd:ro \
#    -v /etc/group:/etc/group:ro \
#    sam3_ft \
#    python sam3/train/train.py -c configs/custom/sam3_ft_ph2.yaml --use-cluster 0 2>&1 | tee logs/sam3_ft_ph2_100_3.log

docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" \
  -e HF_HOME=/workspace/cache/huggingface \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets/isic_10k:/workspace/data \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/sam3:/workspace/sam3 \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/sam3_cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  sam3_ft \
  python sam3/train/train.py -c configs/custom/sam3_ft_isic_10k.yaml --use-cluster 0 2>&1 | tee logs/sam3_ft_isic_10k.log