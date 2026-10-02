#!/bin/bash
# =============================================================================
# wait_gpu_sam3.sh — Aguarda a GPU ficar ociosa e então lança o pipeline SAM 3
# (run_pipeline_sam3.sh) dentro do container ``sam3_ft``.
#
# 1. Consulta ``nvidia-smi`` a cada ``POLL_INTERVAL`` segundos (padrão 5) na
#    GPU ${GPU_DEVICE}.
# 2. A GPU está "livre" quando não roda nenhum processo de computação E
#    memory.used < ``MAX_MEM_MIB`` (1000) E utilization.gpu < ``MAX_UTIL``
#    (10%). Uma falha do ``nvidia-smi`` conta como ocupada.
# 3. O pipeline inicia IMEDIATAMENTE na primeira verificação livre
#    (``CONFIRM_CHECKS=1``). Use ``CONFIRM_CHECKS=N`` para exigir N
#    verificações livres consecutivas (ex.: ignorar o intervalo curto entre
#    dois jobs de outro usuário). Em seguida executa o bloco ``docker run``.
#
# O pipeline é retomável: relançar o mesmo comando continua um estudo
# interrompido em vez de recomeçá-lo (relance SEM --force). O log do terminal
# fica em logs/${PIPELINE_NAME}/terminal_<UTC>.log. Argumentos extras são repassados ao
# run_pipeline_sam3.sh (ex.: ./wait_gpu_sam3.sh --phases "1 2").
# =============================================================================

GPU_DEVICE="${GPU_DEVICE:-0}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_final_v1}"
YOLO26_DATASET="${YOLO26_DATASET:-$(pwd)/../sandbox_yolo26/datasets/isic2018_task1_official}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
CONFIRM_CHECKS="${CONFIRM_CHECKS:-1}"
MAX_MEM_MIB="${MAX_MEM_MIB:-1000}"
MAX_UTIL="${MAX_UTIL:-10}"

gpu_free() {
    local q mem util uuid apps
    STATUS=""
    q=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,uuid --format=csv,noheader,nounits -i "${GPU_DEVICE}" 2>/dev/null) || return 1
    IFS=', ' read -r mem util uuid <<<"${q}"
    [[ "${mem}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]] || return 1
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "${uuid}")
    STATUS="${mem}MiB ${util}% ${apps} proc"
    [ "${apps}" -eq 0 ] && [ "${mem}" -lt "${MAX_MEM_MIB}" ] && [ "${util}" -lt "${MAX_UTIL}" ]
}

echo "Aguardando a GPU ${GPU_DEVICE} ficar livre (consulta a cada ${POLL_INTERVAL}s, ${CONFIRM_CHECKS} verificação(ões))..."
FREE_COUNT=0
LAST_STATE=""
while true; do
    if gpu_free; then
        ((FREE_COUNT++))
        STATE="livre"
    else
        FREE_COUNT=0
        STATE="ocupada"
    fi
    # Registra só as mudanças de estado (a consulta a cada 5 s inundaria o terminal).
    if [ "${STATE}" != "${LAST_STATE}" ]; then
        echo "$(date) | GPU${GPU_DEVICE}: ${STATUS:-nvidia-smi falhou} -> ${STATE}."
        LAST_STATE="${STATE}"
    fi
    [ "${FREE_COUNT}" -ge "${CONFIRM_CHECKS}" ] && break
    sleep "${POLL_INTERVAL}"
done
echo "GPU liberada — iniciando o pipeline SAM 3."

# O dataset YOLO26 (fonte única de verdade) é montado somente-leitura; o
# dataset COCO da Fase 0 é escrito em datasets/isic_2018_task1_sam3. Os pesos
# do SAM 3 vêm do cache local do Hugging Face (sam3_cache/, modo offline).
mkdir -p "logs/${PIPELINE_NAME}"
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
  2>&1 | tee "logs/${PIPELINE_NAME}/terminal_$(date -u +%Y%m%dT%H%M%SZ).log"
