# SAM3 Fine-Tuning on the PH2 Dataset

This repository provides scripts and configuration files for fine-tuning the **SAM3** model on the **PH2 dataset**, focusing on skin lesion segmentation.

The goal is to evaluate how well SAM3 adapts to medical images with limited data and high visual variability.

---

## Overview

The training pipeline includes:

* Loading and preprocessing the PH2 dataset
* Fine-tuning SAM3 for lesion segmentation
* Automatic saving of checkpoints and logs
* Fully reproducible execution via Docker with GPU support

---

## Requirements

* Docker with NVIDIA GPU support
* NVIDIA Container Toolkit installed
* Hugging Face access token
* PH2 dataset available at:

```
./ph2_dataset
```

---

## Expected Project Structure

```
sandbox_sam3/
│
├── logs/               # Training outputs
├── ph2_dataset/        # PH2 dataset
├── sam3/               # Model source code
└── utils/              # Useful scripts
```

---

## Training

Training is executed through Docker to ensure reproducibility.

### Run segmentation training

```bash
docker run --gpus '"device=0"' -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e HUGGING_FACE_HUB_TOKEN=YOUR_HF_TOKEN \
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
  python sam3/train/train.py \
  -c configs/custom/ph2_train_seg.yaml \
  --use-cluster 0
```

---

## Outputs

All outputs are automatically saved to:

```
./logs
```

Each run stores:

* model checkpoints
* validation metrics
* training logs

---

## Running on a Remote Server

### Run training in the background

Create a screen session:

```bash
screen -S training
```

Run the Docker command normally. Detach while keeping the process running:

```
Ctrl + A, then D
```

Reattach later:

```bash
screen -r training
```

---

### Copy results from the server

```bash
rsync -avz --progress -e "ssh -p 13508 -v" antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/sandbox_sam3/logs//ph2_train_seg_500 /home/avmoura_linux/Documents/unb/sandbox_sam3
```

---