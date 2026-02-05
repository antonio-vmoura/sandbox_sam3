# sandbox_sam3

python3.11 --version
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[train, notebooks]"
python sam3/train/train.py -c configs/custom/ph2_train_bbox.yaml --use-cluster 0

mkdir -p logs

# Comando Docker completo
docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e HUGGING_FACE_HUB_TOKEN= \
  -v $(pwd)/ph2_dataset:/workspace/data/ph2 \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/sam3:/workspace/sam3 \
  -v $(pwd)/configs:/workspace/configs \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  sam3-ph2-gpu \
  python sam3/train/train.py -c configs/custom/sam3_ph2_docker.yaml --use-cluster 0