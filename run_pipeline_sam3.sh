#!/usr/bin/env bash
# =============================================================================
# run_pipeline_sam3.sh — Orquestrador do Pipeline SAM 3 
# no dataset ISIC 2018 Task 1 (COCO Format).
# Fases Inclusas: 1 (Baseline), 2 (HPO) e 3 (Otimizado)
# =============================================================================
set -euo pipefail

HOST_LOGS_DIR="$(pwd)/logs"
PIPELINE_NAME="pipeline_sam3_v1"
HOST_PROJECT_DIR="${HOST_LOGS_DIR}/${PIPELINE_NAME}"

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG_DIR="${HOST_PROJECT_DIR}/pipeline_runs/${RUN_TS}"
mkdir -p "${PIPELINE_LOG_DIR}"
PIPELINE_LOG="${PIPELINE_LOG_DIR}/pipeline_sam3.log"

# Define 1 GPU para evitar OutOfMemory com o Ollama
GPU_DEVICE_IDS="0"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${PIPELINE_LOG}"; }

log "=============================================================="
log "SAM 3 ISIC 2018 Task 1 — End-to-End Pipeline (Fases 1 a 3)"
log "=============================================================="

# =============================================================================
# FASE 1: Baseline 
# (Descomente as linhas abaixo quando for rodar o pipeline noturno do zero)
# =============================================================================
log "### Fase 1 — [COMENTADA PARA TESTE]"
docker run --gpus '"device='${GPU_DEVICE_IDS}'"' -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e HUGGING_FACE_HUB_TOKEN="" \
  -e HF_HOME=/workspace/cache/huggingface \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets/isic_2018_task1_coco:/workspace/data \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/sam3:/workspace/sam3 \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/sam3_cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  sam3_ft \
  python sam3/train/train.py \
    -c configs/custom/sam3_phase1_baseline.yaml \
    --use-cluster 0 2>&1 | tee -a "${PIPELINE_LOG}"


# =============================================================================
# FASE 2: Hyperparameter Optimization (Optuna)
# =============================================================================
BEST_HP_YAML="${HOST_PROJECT_DIR}/hpo/best_hyperparameters.yaml"

if [ -f "$BEST_HP_YAML" ]; then
    log "[SKIP] Fase 2 — HPO já foi executado e os hiperparâmetros foram encontrados."
else
    log "### Fase 2 — Iniciando HPO (Optuna)..."
    docker run --gpus '"device='${GPU_DEVICE_IDS}'"' -it --rm \
      --ipc=host \
      --user $(id -u):$(id -g) \
      -e HUGGING_FACE_HUB_TOKEN="" \
      -e HF_HOME=/workspace/cache/huggingface \
      -e TORCH_HOME=/workspace/cache/torch \
      -e HOME=/workspace/cache \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -v $(pwd)/datasets/isic_2018_task1_coco:/workspace/data \
      -v $(pwd)/logs:/workspace/logs \
      -v $(pwd)/sam3:/workspace/sam3 \
      -v $(pwd)/configs:/workspace/configs \
      -v $(pwd)/sam3_cache:/workspace/cache \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      sam3_ft \
      python sam3/train/tune_sam3.py \
        --base_yaml /workspace/sam3/train/configs/custom/sam3_phase1_baseline.yaml \
        --project_dir /workspace/logs/${PIPELINE_NAME}/hpo \
        --trials 15 \
        --epochs 10 \
        --gpus 1 2>&1 | tee -a "${PIPELINE_LOG}"
fi


# =============================================================================
# FASE 3: Treinamento Otimizado (Best Hyperparameters)
# =============================================================================
PHASE3_DIR="${HOST_PROJECT_DIR}/phase3_optimized"
# Ajustado para salvar dentro de sam3/train para o Hydra encontrar
PHASE3_YAML_CONTAINER="/workspace/sam3/train/configs/custom/sam3_phase3_optimized.yaml"

if [ -d "${PHASE3_DIR}/tensorboard" ]; then
    log "[SKIP] Fase 3 — Treinamento Otimizado já foi concluído."
elif [ ! -f "$BEST_HP_YAML" ]; then
    log "[ERRO] Arquivo best_hyperparameters.yaml não encontrado. A Fase 2 falhou ou não concluiu."
else
    log "### Fase 3 — Preparando YAML Otimizado..."
    
    # 1. Executa o gerador de YAML usando a mesma imagem do Docker
    docker run --user $(id -u):$(id -g) --rm \
      -v $(pwd)/logs:/workspace/logs \
      -v $(pwd)/sam3:/workspace/sam3 \
      -v $(pwd)/configs:/workspace/configs \
      sam3_ft \
      python sam3/train/prepare_phase3.py \
        --base_yaml /workspace/sam3/train/configs/custom/sam3_phase1_baseline.yaml \
        --hpo_yaml /workspace/logs/${PIPELINE_NAME}/hpo/best_hyperparameters.yaml \
        --out_yaml ${PHASE3_YAML_CONTAINER} \
        --epochs 40 \
        --log_dir /workspace/logs/${PIPELINE_NAME}/phase3_optimized

    log "### Fase 3 — Iniciando Treinamento Otimizado..."
    
    # 2. Roda o treinamento definitivo
    docker run --gpus '"device='${GPU_DEVICE_IDS}'"' -it --rm \
      --ipc=host \
      --user $(id -u):$(id -g) \
      -e HUGGING_FACE_HUB_TOKEN="" \
      -e HF_HOME=/workspace/cache/huggingface \
      -e TORCH_HOME=/workspace/cache/torch \
      -e HOME=/workspace/cache \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      -v $(pwd)/datasets/isic_2018_task1_coco:/workspace/data \
      -v $(pwd)/logs:/workspace/logs \
      -v $(pwd)/sam3:/workspace/sam3 \
      -v $(pwd)/configs:/workspace/configs \
      -v $(pwd)/sam3_cache:/workspace/cache \
      -v /etc/passwd:/etc/passwd:ro \
      -v /etc/group:/etc/group:ro \
      sam3_ft \
      python sam3/train/train.py \
        -c configs/custom/sam3_phase3_optimized.yaml \
        --use-cluster 0 2>&1 | tee -a "${PIPELINE_LOG}"
fi

log "=============================================================="
log "SAM 3 Fases 1 a 3 Orquestradas com Sucesso."
log "=============================================================="