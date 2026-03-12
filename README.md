# SAM3 Fine-Tuning on the PH2 Dataset

This repository provides scripts and configuration files for fine-tuning the **SAM3** model on the **PH2 dataset**, focusing on skin lesion segmentation.

The goal is to evaluate how well SAM3 adapts to medical images with limited data and high visual variability.

---

## Overview

The training pipeline includes:

* Loading and preprocessing the dataset
* Fine-tuning SAM3 for lesion segmentation
* Automatic saving of checkpoints and logs
* Fully reproducible execution via Docker with GPU support

---

## Requirements

* Docker with NVIDIA GPU support
* NVIDIA Container Toolkit installed
* Hugging Face access token
* Dataset available at:

```
./datasets/<dataset_name>
```

---

## Expected Project Structure

```
sandbox_sam3/
│
├── logs/               # Training outputs
├── dataset/            # Dataset
├── sam3/               # Model source code
└── utils/              # Useful scripts
```

---

## Training

Training is executed through Docker to ensure reproducibility and environment isolation. Don't forget to replace YOUR_HF_TOKEN with your actual Hugging Face token.

### Option A: Run training using ALL available GPUs

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e HUGGING_FACE_HUB_TOKEN=YOUR_HF_TOKEN \
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
```

### Option B: Run training using a SINGLE GPU

```bash
docker run --gpus '"device=0"' -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e HUGGING_FACE_HUB_TOKEN=YOUR_HF_TOKEN \
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
  sam3-ph2-gpu \
  python sam3/train/train.py -c configs/custom/sam3_ft_isic_10k.yaml --use-cluster 0 2>&1 | tee logs/sam3_ft_isic_10k_gpu0.log
```

---

## Outputs

All outputs are automatically saved to:

```
./logs
```

---

## Running on a Remote Server

### Run training in the background

Create a screen session:

```bash
screen -S sam3_ft
```

Run the Docker command normally. Detach while keeping the process running:

```
Ctrl + A, then D
```

Reattach later:

```bash
screen -r sam3_ft
```

---

### Copy results from the server

```bash
rsync -avz --progress -e "ssh -p 13508 -v" antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/sandbox_sam3/logs/isic_train_seg /home/avmoura_linux/Documents/unb/sandbox_sam3
```

---

### Environment Setup

```bash
python3.11 -m venv venv
source venv/bin/activate

```

---

### Hardware Monitoring

```bash

nvidia-smi
nvtop

```

---