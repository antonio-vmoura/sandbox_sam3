# SAM3 Fine-Tuning (PH2 Dataset)

Scripts para treinamento do modelo SAM3 no dataset de lesões de pele PH2.

## Requisitos
* Docker com suporte a NVIDIA GPU.
* Dataset PH2 localizado em `./ph2_dataset`.
* Token do Hugging Face.

---

## Como Executar

### 1. Treinamento de Segmentação
*Foca na geração de máscaras precisas (Batch=1).*

```bash
docker run --gpus '"device=0"' -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e HUGGING_FACE_HUB_TOKEN=SEU_TOKEN_HUGGINGFACE \
  -e HF_HOME=/workspace/cache/huggingface \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/ph2_dataset:/workspace/data/ph2 \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/sam3:/workspace/sam3 \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/sam3_cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  sam3-ph2-gpu \
  python sam3/train/train.py -c configs/custom/ph2_train_seg.yaml --use-cluster 0

```

### 2. Treinamento de Detecção (BBox)

*Foca na predição de caixas delimitadoras.*

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e HUGGING_FACE_HUB_TOKEN=SEU_TOKEN_HUGGINGFACE \
  -e HF_HOME=/workspace/cache/huggingface \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -v $(pwd)/ph2_dataset:/workspace/data/ph2 \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/sam3:/workspace/sam3 \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/sam3_cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  sam3-ph2-gpu \
  python sam3/train/train.py -c configs/custom/ph2_train_bbox.yaml --use-cluster 0

```

## Saídas

Os logs, checkpoints e resultados de validação serão salvos automaticamente na pasta `./logs`.

<!-- rsync -avz --progress -e "ssh -p 13508 -v" antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/sandbox_sam3/logs/ph2_experiment_segmentation_single_gpu /home/avmoura_linux/Documents/unb/sandbox_sam3/ph2_experiment_segmentation_single_gpu -->