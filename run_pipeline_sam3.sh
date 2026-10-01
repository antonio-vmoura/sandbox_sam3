#!/usr/bin/env bash
# =============================================================================
# run_pipeline_sam3.sh — Master orchestrator for the SAM 3 study on the
# ISIC 2018 Task 1 dataset (Meta's official trainer; 5-phase protocol mirrored
# from YOLO26 and the U-Net).
#
#   Phase 0 — COCO dataset          (built from the YOLO26 dataset: identical
#                                    images, splits, masks and CV folds)
#   Phase 1 — Baseline fine-tuning  (base setup + official recipe HPs)
#   Phase 2 — Baseline 5-fold CV    (same folds as YOLO26 / U-Net; test isolated)
#   Phase 3 — HPO                   (Optuna TPE, seeded per proposal,
#                                    resumable; retried on exit code 75)
#   Phase 4 — Optimised fine-tuning (same base setup + Phase 3 HPs)
#   Phase 5 — Test set evaluation   (Baseline AND Optimised; same metrics and
#                                    profiling as YOLO26)
#
# Every phase shares one base setup (official SAM 3 recipe, 1008 px, prompt
# "skin lesion", batch 2, activation checkpointing, FP32, seed 0,
# deterministic kernels except the ViT attention backward), defined once in
# sam3_seg/common.py, so the tuned
# hyperparameters (learning dynamics + augmentation) are the only variable
# between Baseline and Optimised.
#
# Compute budget (SAM 3-specific, disclosed; one FP32 epoch ~ 2 h on a V100S):
#   Phases 1, 2, 4: 30 epochs, patience 30 = no early stopping   (YOLO26 / U-Net: 120 / 120)
#   Phase 3:        10 trials x 10 epochs, patience 10 = no early stopping   (YOLO26 / U-Net: 30 x 30)
#
# Fault tolerance: every step is idempotent and resumable — re-running the
# same command continues where it stopped; interrupted trainings resume from
# the end of their last epoch (exact RNG / optimiser state). --force starts the selected phases over
# (old outputs are moved to *.bak-<UTC>, never deleted).
#
# Runs *inside* the ``sam3_ft`` Docker container (one GPU visible), e.g.:
#
#   docker run --rm --gpus '"device=1"' --ipc=host --user "$(id -u):$(id -g)" \
#     -e HF_HOME=/workspace/cache/huggingface -e HF_HUB_OFFLINE=1 -e HOME=/workspace/cache \
#     -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     -v "$PWD/sam3:/workspace/sam3" -v "$PWD/sam3_seg:/workspace/sam3_seg" \
#     -v "$PWD/sam3_cache:/workspace/cache" -v "$PWD/datasets:/workspace/datasets" \
#     -v "$PWD/logs:/workspace/logs" -v "$PWD/run_pipeline_sam3.sh:/workspace/run_pipeline_sam3.sh:ro" \
#     -v "<sandbox_yolo26>/datasets/isic_2018_task1_yolo26:/workspace/yolo26_dataset:ro" \
#     -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
#     --entrypoint bash sam3_ft /workspace/run_pipeline_sam3.sh --device 0 \
#     --yolo-data /workspace/yolo26_dataset/data.yaml
#
# (/etc/passwd + /etc/group: torch.compile's cache needs the user to resolve.)
# =============================================================================
set -euo pipefail

# ---------- Defaults ---------------------------------------------------------
YOLO_DATA_YAML="${YOLO_DATA_YAML:-/workspace/datasets/isic_2018_task1_yolo26/data.yaml}"
DATA_DIR="${DATA_DIR:-/workspace/datasets/isic_2018_task1_sam3}"
LOGS_ROOT="${LOGS_ROOT:-/workspace/logs}"
PIPELINE_NAME="${PIPELINE_NAME:-pipeline_final_v1}"
if [[ -n "${PROJECT:-}" ]]; then
    PROJECT_FORCED=1
else
    PROJECT_FORCED=0
fi
PROJECT="${PROJECT:-${LOGS_ROOT}/${PIPELINE_NAME}}"
# Single device (SAM 3 is fine-tuned on one GPU).
GPU_DEVICE="${GPU_DEVICE:-0}"
# Phase 5 benchmark device (default: the training device).
BENCH_DEVICE="${BENCH_DEVICE:-}"
MODELS=(sam3)
PHASES=(0 1 2 3 4 5)

# Training budget of Phases 1, 2 and 4 (empty = defaults in common.py:
# 30 epochs / patience 30). Override ONLY for smoke tests.
TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"
TRAIN_PATIENCE="${TRAIN_PATIENCE:-}"

# Phase 2 (CV)
CV_K_FOLDS="${CV_K_FOLDS:-5}"
CV_SEED="${CV_SEED:-0}"

# Phase 3 (HPO) — SAM 3-specific budget (YOLO26 / U-Net: 30 x 30, patience 30).
HPO_ITERATIONS="${HPO_ITERATIONS:-10}"
HPO_EPOCHS_PER_TRIAL="${HPO_EPOCHS_PER_TRIAL:-10}"
HPO_PATIENCE="${HPO_PATIENCE:-10}"   # = epochs per trial: no early stopping
HPO_MAX_RETRIES="${HPO_MAX_RETRIES:-5}"
HPO_RETRY_WAIT="${HPO_RETRY_WAIT:-600}"

# Phase 5
EVAL_PRECISIONS="${EVAL_PRECISIONS:-fp32 fp16}"

FORCE_FLAG=""
DRY_RUN=0
EXIT_GPU_UNAVAILABLE=75

SAM3_SEG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/sam3_seg"
if [[ -d /workspace/sam3_seg ]]; then
    SAM3_SEG_DIR=/workspace/sam3_seg
fi

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --phases "0 1 2 3 4 5"      Subset of phases to run (default: all).
  --yolo-data PATH            YOLO26 data.yaml (source of truth). (env: YOLO_DATA_YAML)
  --data PATH                 Phase 0 COCO dataset directory. (env: DATA_DIR)
  --logs-root PATH            Parent dir of all pipeline runs. (env: LOGS_ROOT)
  --pipeline-name NAME        Sub-directory under LOGS_ROOT. (env: PIPELINE_NAME,
                              default pipeline_final_v1)
  --project PATH              Explicit root (overrides LOGS_ROOT/PIPELINE_NAME).
  --device ID                 Single GPU id. (env: GPU_DEVICE, default 0)
  --bench-device ID           GPU for Phase 5. (env: BENCH_DEVICE, default: --device)
  --epochs INT / --patience INT
                              Smoke-test budget for Phases 1, 2 AND 4 together.
  --force                     Start the selected phases over (old outputs → *.bak-<UTC>).
  --dry-run                   Print the commands without executing them.
  -h, --help                  Show this help.

Environment: YOLO_DATA_YAML, DATA_DIR, LOGS_ROOT, PIPELINE_NAME, PROJECT,
  GPU_DEVICE, BENCH_DEVICE, TRAIN_EPOCHS, TRAIN_PATIENCE, CV_K_FOLDS, CV_SEED,
  HPO_ITERATIONS, HPO_EPOCHS_PER_TRIAL, HPO_PATIENCE, HPO_MAX_RETRIES,
  HPO_RETRY_WAIT, EVAL_PRECISIONS

Exit codes: 0 success; 75 HPO gave up after HPO_MAX_RETRIES GPU failures;
            anything else = exit code of the failing step.
EOF
}

PROJECT_EXPLICIT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phases) read -r -a PHASES <<<"$2"; shift 2 ;;
        --yolo-data) YOLO_DATA_YAML="$2"; shift 2 ;;
        --data) DATA_DIR="$2"; shift 2 ;;
        --logs-root) LOGS_ROOT="$2"; shift 2 ;;
        --pipeline-name) PIPELINE_NAME="$2"; shift 2 ;;
        --project) PROJECT="$2"; PROJECT_EXPLICIT=1; shift 2 ;;
        --device) GPU_DEVICE="$2"; shift 2 ;;
        --bench-device) BENCH_DEVICE="$2"; shift 2 ;;
        --epochs) TRAIN_EPOCHS="$2"; shift 2 ;;
        --patience) TRAIN_PATIENCE="$2"; shift 2 ;;
        --force) FORCE_FLAG="--force"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ "${PROJECT_EXPLICIT}" -eq 0 && "${PROJECT_FORCED}" -eq 0 ]]; then
    PROJECT="${LOGS_ROOT}/${PIPELINE_NAME}"
fi
BENCH_DEVICE="${BENCH_DEVICE:-${GPU_DEVICE}}"
for dev in "${GPU_DEVICE}" "${BENCH_DEVICE}"; do
    if [[ "${dev}" == *,* ]]; then
        echo "[erro] O SAM 3 usa um único dispositivo (ex.: --device 0), não '${dev}'." >&2
        exit 2
    fi
done
for p in "${PHASES[@]}"; do
    if [[ ! "${p}" =~ ^[0-5]$ ]]; then
        echo "[erro] Fase inválida: '${p}'. Use números de 0 a 5." >&2
        exit 2
    fi
done

BUDGET_ARGS=()
[[ -n "${TRAIN_EPOCHS}" ]] && BUDGET_ARGS+=(--epochs "${TRAIN_EPOCHS}")
[[ -n "${TRAIN_PATIENCE}" ]] && BUDGET_ARGS+=(--patience "${TRAIN_PATIENCE}")
FORCE_ARGS=()
[[ -n "${FORCE_FLAG}" ]] && FORCE_ARGS+=("${FORCE_FLAG}")
COMMON_ARGS=(--models "${MODELS[@]}" --data "${DATA_DIR}" --project "${PROJECT}")

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
PIPELINE_LOG_DIR="${PROJECT}/pipeline_runs/${RUN_TS}"
mkdir -p "${PIPELINE_LOG_DIR}"
PIPELINE_LOG="${PIPELINE_LOG_DIR}/pipeline.log"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "${PIPELINE_LOG}"; }

# ---------- GPU sanity check -------------------------------------------------
# Verifica, em <1 s, se o driver NVIDIA e o torch.cuda estão saudáveis antes
# de cada fase (falhar cedo em vez de minutos dentro do treino). Pulada em
# --dry-run e quando o dispositivo é "cpu".
gpu_sanity_check() {
    local tag="$1" dev="${2:-${GPU_DEVICE}}"
    if [[ "${DRY_RUN}" -eq 1 || "${dev}" == "cpu" ]]; then
        log "    [gpu_sanity_check:${tag}] pulado (${dev}; dry-run=${DRY_RUN})"
        return 0
    fi
    log "    [gpu_sanity_check:${tag}] verificando driver NVIDIA e torch.cuda..."
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log "    [gpu_sanity_check:${tag}] FALHOU — 'nvidia-smi' não está no PATH dentro do container."
        return 10
    fi
    if ! nvidia-smi -L >/dev/null 2>&1; then
        log "    [gpu_sanity_check:${tag}] FALHOU — 'nvidia-smi -L' não retornou GPUs (driver caiu?)."
        return 11
    fi
    if ! python - <<'PY' >/dev/null 2>&1
import sys
import torch
sys.exit(0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else 1)
PY
    then
        log "    [gpu_sanity_check:${tag}] FALHOU — torch.cuda.is_available() retornou False."
        return 12
    fi
    log "    [gpu_sanity_check:${tag}] OK — $(nvidia-smi -L | wc -l) GPU(s) visíveis."
    return 0
}

run_cmd() {
    local phase_tag="$1"; shift
    local phase_log="${PIPELINE_LOG_DIR}/${phase_tag}.log"
    log ">>> [${phase_tag}] $*"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "    (dry-run — skipping execution)"
        return 0
    fi
    set +e
    "$@" 2>&1 | tee -a "${phase_log}" | tee -a "${PIPELINE_LOG}"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ $rc -ne 0 ]]; then
        log "<<< [${phase_tag}] FAILED with exit code ${rc}"
        return "${rc}"
    fi
    log "<<< [${phase_tag}] OK"
    return 0
}

run_or_die() {
    local rc=0
    run_cmd "$@" || rc=$?
    if [[ $rc -ne 0 ]]; then
        log "Pipeline aborted at step '$1' (exit ${rc}). Fix the cause and re-run the same command to resume."
        exit "${rc}"
    fi
}

has_phase() {
    local needle="$1"
    for p in "${PHASES[@]}"; do
        [[ "${p}" == "${needle}" ]] && return 0
    done
    return 1
}

require_complete_run() {
    local run_dir="$1" label="$2"
    [[ "${DRY_RUN}" -eq 1 ]] && return 0
    if ! grep -q '"status": "complete"' "${run_dir}/run_state.json" 2>/dev/null; then
        log "[erro] ${label}: run incompleto ou ausente em ${run_dir} — execute a fase correspondente antes."
        exit 3
    fi
}

require_script() {
    if [[ ! -f "${SAM3_SEG_DIR}/$1" ]]; then
        log "[erro] ${SAM3_SEG_DIR}/$1 não existe (ainda não implementado)."
        exit 4
    fi
}

log "=============================================================="
log "SAM 3 (official trainer) ISIC 2018 Task 1 — 5-Phase Pipeline"
log "=============================================================="
log "  yolo data      = ${YOLO_DATA_YAML}"
log "  data           = ${DATA_DIR}"
log "  project        = ${PROJECT}"
log "  device         = ${GPU_DEVICE}   (Phase 5 bench device = ${BENCH_DEVICE})"
log "  phases         = ${PHASES[*]}"
log "  train budget   = ${TRAIN_EPOCHS:-30 (default)} epochs, patience ${TRAIN_PATIENCE:-30 (default)}  [Phases 1, 2, 4]"
log "  cv             = k=${CV_K_FOLDS}, seed=${CV_SEED}"
log "  hpo            = Optuna TPE, trials=${HPO_ITERATIONS}, ep/trial=${HPO_EPOCHS_PER_TRIAL}, patience=${HPO_PATIENCE}, retries=${HPO_MAX_RETRIES} x ${HPO_RETRY_WAIT}s"
log "  precisions     = ${EVAL_PRECISIONS}  [Phase 5]"
log "  force          = ${FORCE_FLAG:-<off>}"
log "  sam3_seg_dir   = ${SAM3_SEG_DIR}"
if [[ -n "${TRAIN_EPOCHS}${TRAIN_PATIENCE}" ]]; then
    log "  [aviso] orçamento de treino diferente do protocolo (30/30) — use apenas para smoke tests."
fi
log "--------------------------------------------------------------"

# ---------- Phase 0 — COCO dataset --------------------------------------------
if has_phase 0; then
    log ""
    log "### Phase 0 — SAM 3 COCO dataset from the YOLO26 dataset (idempotent)"
    run_or_die phase0 python "${SAM3_SEG_DIR}/prepare_dataset.py" \
        --yolo-data "${YOLO_DATA_YAML}" --out "${DATA_DIR}" "${FORCE_ARGS[@]}"
fi

# ---------- Phase 1 — Baseline training -------------------------------------
if has_phase 1; then
    log ""
    log "### Phase 1 — Baseline fine-tuning (base setup + official recipe HPs)"
    gpu_sanity_check phase1 || exit $?
    run_or_die phase1 python "${SAM3_SEG_DIR}/train_baseline_sam3.py" \
        "${COMMON_ARGS[@]}" --device "${GPU_DEVICE}" "${BUDGET_ARGS[@]}" "${FORCE_ARGS[@]}"
    run_or_die phase1_collect python "${SAM3_SEG_DIR}/collect_phase_metrics_sam3.py" \
        --phase phase1 --models "${MODELS[@]}" --project "${PROJECT}"
fi

# ---------- Phase 2 — Baseline cross-validation -----------------------------
if has_phase 2; then
    log ""
    log "### Phase 2 — Baseline ${CV_K_FOLDS}-fold CV on train+val (seed=${CV_SEED}; same folds as YOLO26 / U-Net; test isolated)"
    gpu_sanity_check phase2 || exit $?
    run_or_die phase2 python "${SAM3_SEG_DIR}/train_cv_sam3.py" \
        --protocol baseline "${COMMON_ARGS[@]}" --device "${GPU_DEVICE}" \
        --k-folds "${CV_K_FOLDS}" --seed "${CV_SEED}" "${BUDGET_ARGS[@]}" "${FORCE_ARGS[@]}"
    run_or_die phase2_consolidate python "${SAM3_SEG_DIR}/consolidate_cv_results_sam3.py" \
        --protocol baseline --models "${MODELS[@]}" --project "${PROJECT}"
fi

# ---------- Phase 3 — HPO (fault-tolerant, retried on exit 75) --------------
if has_phase 3; then
    log ""
    log "### Phase 3 — HPO (Optuna TPE, ${HPO_ITERATIONS} trials x ${HPO_EPOCHS_PER_TRIAL} ep, seeded, resumable)"
    HPO_FORCE_ARGS=("${FORCE_ARGS[@]}")
    max_attempts=$((HPO_MAX_RETRIES + 1))
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        rc=0
        if ! gpu_sanity_check "phase3#${attempt}"; then
            rc=${EXIT_GPU_UNAVAILABLE}
        else
            run_cmd phase3 python "${SAM3_SEG_DIR}/tune_sam3.py" \
                "${COMMON_ARGS[@]}" --device "${GPU_DEVICE}" \
                --iterations "${HPO_ITERATIONS}" \
                --epochs "${HPO_EPOCHS_PER_TRIAL}" \
                --patience "${HPO_PATIENCE}" \
                "${HPO_FORCE_ARGS[@]}" || rc=$?
        fi
        HPO_FORCE_ARGS=()   # --force only on the first attempt; retries resume
        if [[ ${rc} -eq 0 ]]; then
            break
        fi
        if [[ ${rc} -ne ${EXIT_GPU_UNAVAILABLE} ]]; then
            log "Pipeline aborted at step 'phase3' (exit ${rc}, not a GPU failure). Fix the cause and re-run to resume."
            exit "${rc}"
        fi
        if [[ ${attempt} -ge ${max_attempts} ]]; then
            log "[erro] Phase 3: GPU indisponível após ${max_attempts} tentativa(s). Recupere o driver e re-execute — a busca continua do último trial."
            exit "${EXIT_GPU_UNAVAILABLE}"
        fi
        log "    [phase3] GPU indisponível (exit ${rc}) — tentativa ${attempt}/${max_attempts}; nova tentativa em ${HPO_RETRY_WAIT}s (retoma do último trial)."
        [[ "${DRY_RUN}" -eq 1 ]] || sleep "${HPO_RETRY_WAIT}"
    done
    log "### Phase 3 — Validating HPO outputs (completeness + degenerate trials)"
    run_or_die phase3_validate python "${SAM3_SEG_DIR}/check_hpo_validity.py" \
        --project "${PROJECT}" --models "${MODELS[@]}" --iterations "${HPO_ITERATIONS}"
fi

# ---------- Phase 4 — Optimised fine-tuning ---------------------------------
if has_phase 4; then
    log ""
    log "### Phase 4 — Optimised fine-tuning with the Phase 3 hyperparameters"
    gpu_sanity_check phase4 || exit $?
    run_or_die phase4 python "${SAM3_SEG_DIR}/train_optimized_sam3.py" \
        "${COMMON_ARGS[@]}" --device "${GPU_DEVICE}" "${BUDGET_ARGS[@]}" "${FORCE_ARGS[@]}"
    run_or_die phase4_collect python "${SAM3_SEG_DIR}/collect_phase_metrics_sam3.py" \
        --phase phase4 --models "${MODELS[@]}" --project "${PROJECT}"
fi

# ---------- Phase 5 — Test set inference & profiling ------------------------
if has_phase 5; then
    log ""
    log "### Phase 5 — Test set evaluation of Baseline AND Optimised (device ${BENCH_DEVICE})"
    for script in evaluate_test_set.py benchmark_efficiency.py build_final_report.py; do
        require_script "${script}"
    done
    for m in "${MODELS[@]}"; do
        require_complete_run "${PROJECT}/phase1_baseline/${m}_baseline" "Baseline ${m}"
        require_complete_run "${PROJECT}/phase4_optimized/${m}_optimized" "Optimised ${m}"
    done
    gpu_sanity_check phase5 "${BENCH_DEVICE}" || exit $?
    read -r -a PRECISIONS <<<"${EVAL_PRECISIONS}"
    run_or_die phase5_accuracy python "${SAM3_SEG_DIR}/evaluate_test_set.py" \
        "${COMMON_ARGS[@]}" --variants baseline optimized --precisions "${PRECISIONS[@]}" \
        --device "${BENCH_DEVICE}" "${FORCE_ARGS[@]}"
    run_or_die phase5_efficiency python "${SAM3_SEG_DIR}/benchmark_efficiency.py" \
        "${COMMON_ARGS[@]}" --variants baseline optimized --precisions "${PRECISIONS[@]}" \
        --device "${BENCH_DEVICE}" "${FORCE_ARGS[@]}"
    run_or_die phase5_report python "${SAM3_SEG_DIR}/build_final_report.py" \
        --models "${MODELS[@]}" --project "${PROJECT}"
fi

log ""
log "=============================================================="
log "Pipeline finished. Per-phase logs: ${PIPELINE_LOG_DIR}/"
log "Consolidated artefacts: ${PROJECT}/summary/"
log "=============================================================="
