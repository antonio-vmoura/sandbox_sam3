#!/bin/bash
# =============================================================================
# wait_gpu_sam3.sh — Aguarda a GPU ficar ociosa e então lança o pipeline SAM 3
# (run_pipeline_sam3.sh) dentro do container ``sam3_ft``.
#
# 1. Consulta ``nvidia-smi`` uma vez por minuto na GPU ${GPU_DEVICE}.
# 2. A GPU é considerada ociosa quando memory.used < 1000 MiB E
#    utilization.gpu < 10%.
# 3. Após ``REQUIRED_IDLE_MINUTES`` verificações ociosas consecutivas, executa
#    o bloco ``docker run`` abaixo.
#
# O pipeline é retomável: relançar o mesmo comando continua um estudo
# interrompido em vez de recomeçá-lo. Argumentos extras são repassados ao
# run_pipeline_sam3.sh (ex.: ./wait_gpu_sam3.sh --phases "1 2").
# =============================================================================

GPU_DEVICE="${GPU_DEVICE:-0}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_final_v1}"
YOLO26_DATASET="${YOLO26_DATASET:-$(pwd)/../sandbox_yolo26/datasets/isic_2018_task1_yolo26}"
CHECK_INTERVAL=60
REQUIRED_IDLE_MINUTES=3
IDLE_COUNT=0

echo "Aguardando a GPU ${GPU_DEVICE} ficar ociosa por ${REQUIRED_IDLE_MINUTES} minuto(s)..."

while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_DEVICE}")
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU_DEVICE}")
    if [ "$MEM" -lt 1000 ] && [ "$UTIL" -lt 10 ]; then
        ((IDLE_COUNT++))
        echo "$(date) | GPU${GPU_DEVICE}: ${MEM}MiB ${UTIL}% -> ociosa há $IDLE_COUNT minuto(s)."
        if [ "$IDLE_COUNT" -ge "$REQUIRED_IDLE_MINUTES" ]; then
            echo "GPU liberada — iniciando o pipeline SAM 3."
            break
        fi
    else
        if [ "$IDLE_COUNT" -gt 0 ]; then
            echo "$(date) | Atividade detectada — zerando contador de ociosidade."
        else
            echo "$(date) | GPU${GPU_DEVICE}: ${MEM}MiB ${UTIL}% -> ocupada."
        fi
        IDLE_COUNT=0
    fi
    sleep $CHECK_INTERVAL
done

# O dataset YOLO26 (fonte única de verdade) é montado somente-leitura; o
# dataset COCO da Fase 0 é escrito em datasets/isic_2018_task1_sam3. Os pesos
# do SAM 3 vêm do cache local do Hugging Face (sam3_cache/, modo offline).
mkdir -p logs
docker run --gpus "\"device=${GPU_DEVICE}\"" --rm --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/workspace/cache/huggingface -e HF_HUB_OFFLINE=1 -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e GPU_DEVICE=0 -e PIPELINE_NAME="${PIPELINE_NAME}" \
  -v "$(pwd)/sam3:/workspace/sam3" -v "$(pwd)/sam3_seg:/workspace/sam3_seg" \
  -v "$(pwd)/sam3_cache:/workspace/cache" -v "$(pwd)/datasets:/workspace/datasets" \
  -v "$(pwd)/logs:/workspace/logs" \
  -v "$(pwd)/run_pipeline_sam3.sh:/workspace/run_pipeline_sam3.sh:ro" \
  -v "${YOLO26_DATASET}:/workspace/yolo26_dataset:ro" \
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
  --entrypoint bash sam3_ft \
  /workspace/run_pipeline_sam3.sh --yolo-data /workspace/yolo26_dataset/data.yaml "$@" \
  2>&1 | tee "logs/${PIPELINE_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
