import json
import numpy as np
from pycocotools import mask as maskUtils

# 1. Configurar os caminhos dos arquivos (ajuste conforme o seu diretório)
gt_file = '/home/antoniovinicius/projects/sandbox_sam3/datasets/isic_2018_task1_coco/valid/_annotations.coco.json'
pred_file = '/home/antoniovinicius/projects/sandbox_sam3/logs/pipeline_sam3_v1/phase1_baseline/dumps/coco_predictions_segm.json'

print("Carregando arquivos JSON...")
# Carregar as Anotações Ground Truth (GT)
with open(gt_file, 'r') as f:
    gt_data = json.load(f)

# Criar dicionário para acessar as máscaras GT por image_id
gt_masks_dict = {}
for ann in gt_data['annotations']:
    img_id = ann['image_id']
    if img_id not in gt_masks_dict:
        gt_masks_dict[img_id] = []
    gt_masks_dict[img_id].append(ann)

# Obter informações das imagens (altura e largura)
img_info_dict = {img['id']: img for img in gt_data['images']}

# Carregar as Predições do Modelo SAM 3
with open(pred_file, 'r') as f:
    pred_data = json.load(f)

# Agrupar predições por image_id
pred_masks_dict = {}
for pred in pred_data:
    img_id = pred['image_id']
    if img_id not in pred_masks_dict:
        pred_masks_dict[img_id] = []
    pred_masks_dict[img_id].append(pred)

def calculate_metrics(gt_mask, pred_mask):
    """Calcula Dice e Jaccard (IoU) entre duas máscaras binárias."""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    
    if union == 0:
        iou = 1.0 if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0 else 0.0
    else:
        iou = intersection / union
        
    dice = (2.0 * intersection) / (np.sum(gt_mask) + np.sum(pred_mask)) if (np.sum(gt_mask) + np.sum(pred_mask)) > 0 else 1.0
    
    return float(iou), float(dice)

def get_binary_mask(segm, h, w):
    """Converte diferentes formatos COCO (Polygon, Uncompressed RLE, Compressed RLE) para máscara binária Numpy."""
    if isinstance(segm, list):
        # Formato de Polígonos
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(segm, dict):
        if isinstance(segm.get('counts'), list):
            # Uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # Compressed RLE (como o SAM e seu GT salvam)
            rle = segm.copy()
            if isinstance(rle['counts'], str):
                rle['counts'] = rle['counts'].encode('utf-8')
    else:
        raise ValueError(f"Formato de segmentação desconhecido: {type(segm)}")
        
    # Decodificar para máscara binária numpy
    m = maskUtils.decode(rle)
    if len(m.shape) > 2:
        m = np.max(m, axis=2)
    return m

iou_list = []
dice_list = []

print("Calculando métricas para cada imagem...")
# 3. Processar cada imagem e calcular as métricas
for img_id, gt_anns in gt_masks_dict.items():
    if img_id not in pred_masks_dict:
        # Se a imagem não tem predição
        iou_list.append(0.0)
        dice_list.append(0.0)
        continue 
        
    img_info = img_info_dict[img_id]
    h, w = img_info['height'], img_info['width']
    
    # Criar a máscara GT binária combinando todas as anotações da imagem
    gt_mask_combined = np.zeros((h, w), dtype=np.uint8)
    for ann in gt_anns:
        m = get_binary_mask(ann['segmentation'], h, w)
        gt_mask_combined = np.maximum(gt_mask_combined, m)
        
    # Criar a máscara de Predição
    preds_for_img = pred_masks_dict[img_id]
    # Pega a predição com o maior score para a imagem
    best_pred = max(preds_for_img, key=lambda x: x['score'])
    
    pred_mask_combined = get_binary_mask(best_pred['segmentation'], h, w)
    
    # Calcular métricas e adicionar à lista
    iou, dice = calculate_metrics(gt_mask_combined, pred_mask_combined)
    iou_list.append(iou)
    dice_list.append(dice)

# 4. Calcular a média final
mean_iou = np.mean(iou_list)
mean_dice = np.mean(dice_list)

print("\n" + "="*30)
print(f"--- Métricas Finais SAM 3 ---")
print("="*30)
print(f"Jaccard (IoU) Médio : {mean_iou:.4f}")
print(f"Dice Médio          : {mean_dice:.4f}")
print("="*30)

# Salvar em um arquivo de texto
output_file = 'sam3_dice_jaccard_metrics.txt'
with open(output_file, 'w') as out_f:
    out_f.write(f"Jaccard (IoU) Médio: {mean_iou:.4f}\n")
    out_f.write(f"Dice Médio: {mean_dice:.4f}\n")
print(f"Métricas salvas com sucesso em: {output_file}")