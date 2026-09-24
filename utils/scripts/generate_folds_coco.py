import json
import os
from pathlib import Path
from sklearn.model_selection import KFold

def split_coco_dataset(coco_json_path, output_dir, n_folds=5):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    # Validação Cruzada: 5 Splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"-> Gerando Fold {fold}...")
        
        train_images = [images[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        
        train_img_ids = {img['id'] for img in train_images}
        val_img_ids = {img['id'] for img in val_images}
        
        train_anns = [ann for ann in annotations if ann['image_id'] in train_img_ids]
        val_anns = [ann for ann in annotations if ann['image_id'] in val_img_ids]
        
        # Cria a estrutura de pastas do Fold exigida pelo SAM 3
        fold_dir = out_path / f"fold_{fold}"
        (fold_dir / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "valid").mkdir(parents=True, exist_ok=True)
        
        # Salva o COCO de Treino
        with open(fold_dir / "train" / "_annotations.coco.json", 'w') as f:
            json.dump({"images": train_images, "annotations": train_anns, "categories": categories}, f)
            
        # Salva o COCO de Validação
        with open(fold_dir / "valid" / "_annotations.coco.json", 'w') as f:
            json.dump({"images": val_images, "annotations": val_anns, "categories": categories}, f)
            
    print(f"✅ Particionamento COCO concluído em: {output_dir}")

if __name__ == "__main__":
    # Caminho do JSON contendo todo o seu dataset de treino (ISIC)
    ORIGINAL_JSON = "datasets/isic_2018_task1_coco/train/_annotations.coco.json"
    OUTPUT_FOLDS = "datasets/isic_2018_task1_coco_folds"
    split_coco_dataset(ORIGINAL_JSON, OUTPUT_FOLDS)