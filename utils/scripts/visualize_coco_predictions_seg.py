import json
import cv2
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from pycocotools import mask as mask_utils

# ================= CONFIGURAÇÃO =================
LOCAL_PATH = "/home/avmoura_linux/Documents/unb/sandbox_sam3"

CONFIG = {
    "images_dir": Path(f"{LOCAL_PATH}/ph2_dataset/test/"),
    # "pred_json": Path(f"{LOCAL_PATH}/logs/old_logs/detection/dumps/ph2/coco_predictions_bbox.json"),
    "pred_json": Path(f"{LOCAL_PATH}/logs/ph2_train_seg/dumps/ph2/coco_predictions_segm.json"),
    "gt_json": Path(f"{LOCAL_PATH}/ph2_dataset/test/_annotations.coco.json"),
    "output_dir": Path(f"{LOCAL_PATH}/logs/visual_results"),
    "score_threshold": 0.45,
    "mask_alpha": 0.45, # Transparência da máscara (0.0 a 1.0)
    "valid_extensions": {'.png', '.jpg', '.jpeg', '.bmp'}
}
# ================================================

def load_json_data(pred_path: Path, gt_path: Path) -> Tuple[Dict[int, List[Dict]], Dict[int, str]]:
    """Carrega predições e cria o mapa de ID -> Nome do Arquivo usando o GT."""
    print(f"--> Carregando Predições: {pred_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"JSON de predição não encontrado: {pred_path}")
    
    with open(pred_path, 'r') as f:
        preds = json.load(f)
    
    # Agrupar predições por imagem
    preds_map = {}
    for p in preds:
        preds_map.setdefault(p['image_id'], []).append(p)

    print(f"--> Carregando Ground Truth (para mapeamento): {gt_path}")
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        # Cria dicionário: {image_id: file_name}
        id_to_filename = {img['id']: img['file_name'] for img in gt['images']}
        print(f"Mapeamento criado para {len(id_to_filename)} imagens.")
    else:
        print("[AVISO] GT não encontrado. Tentaremos usar nomes de arquivo ordenados (arriscado).")
        id_to_filename = None

    return preds_map, id_to_filename

def generate_random_color():
    """Gera uma cor aleatória vibrante (BGR)."""
    return [random.randint(50, 255) for _ in range(3)]

def apply_mask(image: np.ndarray, mask: np.ndarray, color: List[int], alpha: float):
    """Aplica a máscara colorida com transparência sobre a imagem original."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def draw_visualizations(img: np.ndarray, predictions: List[Dict], threshold: float) -> Tuple[int, np.ndarray]:
    """Desenha máscaras e bounding boxes com suporte a coordenadas normalizadas."""
    count = 0
    img_h, img_w = img.shape[:2]  # Obtém dimensões reais da imagem carregada
    predictions.sort(key=lambda x: x['score']) 

    for pred in predictions:
        score = pred['score']
        if score < threshold:
            continue

        color = generate_random_color()
        
        # --- 1. Desenhar Máscara (Segmentação) ---
        if 'segmentation' in pred:
            try:
                rle = pred['segmentation']
                binary_mask = mask_utils.decode(rle)
                
                # IMPORTANTE: Se a máscara (ex: 640x640) for diferente da imagem (ex: 768x560), 
                # precisamos redimensionar para evitar erros de shape no np.where
                if binary_mask.shape[:2] != (img_h, img_w):
                    binary_mask = cv2.resize(binary_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                img = apply_mask(img, binary_mask, color, CONFIG["mask_alpha"])
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
            except Exception as e:
                print(f"    [ERRO] Falha ao processar máscara: {e}")

        # --- 2. Desenhar Bounding Box (Correção de Escala) ---
        bbox = pred['bbox']
        
        # Verifica se as coordenadas estão normalizadas (ex: 0.27 em vez de 180 pixels)
        # Se o valor for menor que 2, assumimos que está normalizado
        if all(v <= 1.1 for v in bbox):
            x = int(bbox[0] * img_w)
            y = int(bbox[1] * img_h)
            w = int(bbox[2] * img_w)
            h = int(bbox[3] * img_h)
        else:
            x, y, w, h = map(int, bbox)
        
        # Desenha o Bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # --- 3. Desenhar Texto ---
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x, y - 20), (x + tw, y), color, -1)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        count += 1
    return count, img

def main():
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    try:
        preds_map, id_to_filename = load_json_data(CONFIG["pred_json"], CONFIG["gt_json"])
    except Exception as e:
        print(f"Erro fatal: {e}")
        return

    print(f"--> Iniciando processamento visual (Threshold: {CONFIG['score_threshold']})...")
    saved_count = 0

    # Iterar sobre as imagens que têm predições
    for img_id, predictions in preds_map.items():
        
        # Descobrir o nome do arquivo
        if id_to_filename:
            # Método Seguro: Usar o ID do GT
            if img_id not in id_to_filename:
                continue
            filename = id_to_filename[img_id]
            img_path = CONFIG["images_dir"] / filename
        else:
            # Fallback (Arriscado): Tentar achar o arquivo na pasta
            # Isso assume que você sabe qual arquivo é qual ID, o que raramente é verdade sem o GT
            print("    [PULANDO] Sem arquivo GT para mapear IDs para nomes de arquivo.")
            break

        if not img_path.exists():
            # Tenta verificar se o nome no JSON é apenas o nome base ou caminho relativo
            img_path = CONFIG["images_dir"] / Path(filename).name
            if not img_path.exists():
                print(f"    [AVISO] Imagem não encontrada em disco: {img_path}")
                continue

        # Carregar Imagem
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Desenhar
        drawn_count, visual_img = draw_visualizations(img, predictions, CONFIG["score_threshold"])

        if drawn_count > 0:
            output_name = f"vis_{img_path.stem}.jpg"
            output_path = CONFIG["output_dir"] / output_name
            cv2.imwrite(str(output_path), visual_img)
            saved_count += 1
            print(f"    Salvo: {output_name} ({drawn_count} objetos)")

    print("-" * 30)
    print(f"RESUMO: {saved_count} imagens salvas em '{CONFIG['output_dir']}'")

if __name__ == "__main__":
    main()