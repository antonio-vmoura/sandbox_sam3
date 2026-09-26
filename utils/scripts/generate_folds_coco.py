import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import KFold

def merge_and_split_coco(train_json, valid_json, train_dir, valid_dir, output_dir, n_folds=5):
    # 1. Ler ficheiros JSON de Treino e Validação
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    with open(valid_json, 'r') as f:
        valid_data = json.load(f)
        
    # 2. Fundir as imagens e anotações num único pool (equivalente à Fase 4 do YOLO/U-Net)
    images = train_data['images'] + valid_data['images']
    annotations = train_data['annotations'] + valid_data['annotations']
    categories = train_data['categories']
    
    # Mapeamento para saber de onde copiar o ficheiro físico
    img_source_map = {}
    train_source = Path(train_dir).resolve()
    valid_source = Path(valid_dir).resolve()
    
    for img in train_data['images']:
        img_source_map[img['file_name']] = train_source
    for img in valid_data['images']:
        img_source_map[img['file_name']] = valid_source
    
    # 3. K-Fold Determinístico (random_state=0 para paridade com YOLO e U-Net)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
        print(f"-> A processar Fold {fold} e a copiar imagens...")
        
        train_images = [images[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        
        train_img_ids = {img['id'] for img in train_images}
        val_img_ids = {img['id'] for img in val_images}
        
        train_anns = [ann for ann in annotations if ann['image_id'] in train_img_ids]
        val_anns = [ann for ann in annotations if ann['image_id'] in val_img_ids]
        
        fold_dir = out_path / f"fold_{fold}"
        train_fold_dir = fold_dir / "train"
        valid_fold_dir = fold_dir / "valid"
        
        train_fold_dir.mkdir(parents=True, exist_ok=True)
        valid_fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar os JSONs COCO para este fold
        with open(train_fold_dir / "_annotations.coco.json", 'w') as f:
            json.dump({"images": train_images, "annotations": train_anns, "categories": categories}, f)
            
        with open(valid_fold_dir / "_annotations.coco.json", 'w') as f:
            json.dump({"images": val_images, "annotations": val_anns, "categories": categories}, f)
            
        # Copiar fisicamente as imagens de treino
        for img in train_images:
            img_name = img['file_name']
            src_img = img_source_map[img_name] / img_name
            dst_img = train_fold_dir / img_name
            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                
        # Copiar fisicamente as imagens de validação
        for img in val_images:
            img_name = img['file_name']
            src_img = img_source_map[img_name] / img_name
            dst_img = valid_fold_dir / img_name
            if src_img.exists() and not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                
    print(f"✅ Folds unificados gerados com sucesso em: {output_dir}")

if __name__ == "__main__":
    TRAIN_JSON = "datasets/isic_2018_task1_coco/train/_annotations.coco.json"
    TRAIN_DIR = "datasets/isic_2018_task1_coco/train"
    
    VALID_JSON = "datasets/isic_2018_task1_coco/valid/_annotations.coco.json"
    VALID_DIR = "datasets/isic_2018_task1_coco/valid"
    
    OUTPUT_FOLDS = "datasets/isic_2018_task1_coco_folds"
    
    merge_and_split_coco(TRAIN_JSON, VALID_JSON, TRAIN_DIR, VALID_DIR, OUTPUT_FOLDS)