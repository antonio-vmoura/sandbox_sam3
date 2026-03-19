import argparse
import base64
import glob
import json
import os
import shutil
import sys
import zlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as pymask


class NpEncoder(json.JSONEncoder):
    """
    Classe auxiliar para o serializador JSON lidar com tipos de dados do NumPy.
    Evita erros ao salvar valores numéricos extraídos de matrizes.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_categories_from_meta(meta_json_path: str) -> dict:
    """
    Lê o arquivo meta.json e mapeia os nomes das classes para IDs numéricos.
    Ignora a classe de fundo ('bg').
    """
    with open(meta_json_path, 'r', encoding='utf-8') as fs:
        json_meta = json.load(fs)
    
    classes = [clss['title'] for clss in json_meta['classes'] if clss['title'] != 'bg']
    map_categories = {c: i for i, c in enumerate(classes)}
    return map_categories


def get_all_ann_file(base_dir: str):
    """
    Busca todos os arquivos .json de anotação no diretório especificado
    e retorna uma lista com os nomes originais das imagens e o conteúdo dos JSONs.
    """
    all_ann_files = glob.glob(os.path.join(base_dir, "*.json"))
    all_fname_img = [fname[:-5] for fname in all_ann_files] # Remove o '.json' do final
    
    all_json_ann = []
    for json_path in all_ann_files:
        with open(json_path, 'r', encoding='utf-8') as fs:
            json_suprv = json.load(fs)
        all_json_ann.append(json_suprv)
        
    return all_fname_img, all_json_ann


def decode_bitmap_mask(s: str) -> np.ndarray:
    """
    Descompacta a string em base64 e zlib fornecida pelo formato Supervisely
    e a converte em uma matriz binária (numpy array) representando a máscara.
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)

    imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    
    # Trata imagens com 4 canais (RGBA) ou máscaras 2D planas
    if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
        mask = imdecoded[:, :, 3].astype(bool)  
    elif len(imdecoded.shape) == 2:
        mask = imdecoded.astype(bool)  
    else:
        raise RuntimeError('Formato interno da máscara incorreto ou não suportado.')
    return mask


def convert_single_image(idimg: int, fname_img: str, json_suprv: dict, map_category: dict, only_img_name: bool = True, start_annotation_id: int = 0):
    """
    Processa uma única imagem e extrai suas anotações geométricas 
    (convertendo de bitmap/polygon para o formato RLE exigido pelo COCO).
    """
    # Define os metadados básicos da imagem
    image_base = {
        "id": idimg,
        "width": json_suprv['size']['width'],
        "height": json_suprv['size']['height'],
        "file_name": Path(fname_img).name if only_img_name else fname_img,
        "license": 1,
        "date_captured": ""
    }

    # Filtra objetos descartando o fundo (background)
    objects = [obj for obj in json_suprv['objects'] if obj['classTitle'] != 'bg']

    h, w = json_suprv['size']['height'], json_suprv['size']['width']
    rles = []
    boxes = []
    areas = []
    
    for instance in objects:
        # Processamento para máscaras em formato de imagem compactada (ISIC 2018 usa este)
        if instance['geometryType'] == 'bitmap':
            full_mask_img = np.zeros((h, w), dtype=bool)
            inst_mask = decode_bitmap_mask(instance['bitmap']['data'])
            inst_w, inst_h = inst_mask.shape
            
            # Posiciona a máscara recortada dentro das dimensões originais da imagem
            id_h, id_w = instance['bitmap']['origin']
            full_mask_img[id_w:id_w+inst_w, id_h:id_h+inst_h] = inst_mask
            
            # Codifica a matriz booleana para RLE (Run-Length Encoding)
            rle = pymask.encode(np.asfortranarray(full_mask_img)) 
            rle['counts'] = rle['counts'].decode('ascii')
            
            rles.append(rle)
            boxes.append(pymask.toBbox(rle))
            areas.append(pymask.area(rle))
            
        # Processamento alternativo para coordenadas de polígonos simples
        elif instance['geometryType'] == 'polygon':
            full_mask_img = np.zeros((h, w))
            interior_mask = np.zeros((h, w))
            
            pts = np.array(instance['points']['exterior']).astype(int)
            cv2.fillPoly(full_mask_img, [pts], 1)
            
            # Subtrai o interior do polígono (buracos na máscara, se houver)
            if len(instance['points']['interior']) > 0:
                for inst_pts in instance['points']['interior']:
                    pts = np.array(inst_pts).astype(int)
                    cv2.fillPoly(interior_mask, [pts], 1) 
                    
            full_mask_img -= interior_mask
            
            rle = pymask.encode(np.asfortranarray(full_mask_img.astype(bool))) 
            rle['counts'] = rle['counts'].decode('ascii')
            
            rles.append(rle)
            boxes.append(pymask.toBbox(rle))
            areas.append(pymask.area(rle))

    # Constrói o dicionário de anotações para o formato COCO
    ann = [
        {
            "id": start_annotation_id + i,
            "image_id": idimg,
            "segmentation": segm,
            "area": area,
            "bbox": bbox,
            "category_id": map_category[obj['classTitle']],
            "iscrowd": 0
        }
        for i, (obj, segm, bbox, area) in enumerate(zip(objects, rles, boxes, areas))
    ]

    return image_base, ann


def convert_supervisely_to_coco(meta_path: str, ann_base_dir: str = './ds/ann/', save_as: str = None, only_img_name: bool = True):
    """
    Gerencia a conversão de todas as imagens de um diretório e gera o dicionário
    final estruturado no padrão oficial do COCO Dataset.
    """
    ann_fnames, ann_jsons = get_all_ann_file(ann_base_dir)
    map_category = get_categories_from_meta(meta_path)

    # Formata as categorias para o cabeçalho do COCO
    catg_repr = [
        {"id": v, "name": k, "supercategory": "type"} 
        for k, v in map_category.items()
    ]

    out_cnv_imgs = [
        convert_single_image(id_img, ann_fnames[id_img], ann_jsons[id_img], map_category, only_img_name)
        for id_img in range(len(ann_fnames))
    ]
    
    # Separa as imagens das anotações
    images_repr = [o[0] for o in out_cnv_imgs]
    ann_repr = [o[1] for o in out_cnv_imgs]

    # Achata a lista de listas de anotações em uma lista simples (1D)
    ann_repr_flatten = [inner for lst in ann_repr for inner in lst]

    # Ajusta os IDs das anotações para serem sequenciais e únicos
    for i, ann in enumerate(ann_repr_flatten):
        ann['id'] = i

    # Monta a estrutura raiz do JSON
    coco_fmt = {
        "info": {
            "year": datetime.now().strftime('%Y'),
            "version": "1.0",
            "description": "Dataset convertido para pesquisa acadêmica",
            "contributor": "Supervisely to COCO Converter",
            "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        },
        "images": images_repr,
        "annotations": ann_repr_flatten,
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": catg_repr
    }

    # Salva o resultado no disco, se um caminho for providenciado
    if save_as:
        with open(save_as, 'w', encoding='utf-8') as fp:
            json.dump(coco_fmt, fp, cls=NpEncoder)
            
    return coco_fmt


def process_full_dataset(input_dir: str, output_dir: str):
    """
    Lê a estrutura completa do Supervisely e cria uma nova estrutura de diretórios
    focada no padrão COCO (imagens e anotações juntas).
    """
    meta_path = os.path.join(input_dir, 'meta.json')
    if not os.path.exists(meta_path):
        print(f"Erro: meta.json não encontrado em {input_dir}")
        sys.exit(1)

    # Mapeamento do nome das pastas (De Supervisely para o padrão COCO/YOLO)
    splits = {
        'train': 'train',
        'val': 'valid',
        'test': 'test'
    }

    os.makedirs(output_dir, exist_ok=True)

    for sup_split, coco_split in splits.items():
        ann_dir = os.path.join(input_dir, sup_split, 'ann')
        img_dir = os.path.join(input_dir, sup_split, 'img')
        
        # Pula processamento caso a pasta (ex: test/ann) não exista no dataset original
        if not os.path.exists(ann_dir) or not os.path.exists(img_dir):
            print(f"Aviso: Pasta de anotação ou imagem ausente para o split '{sup_split}'. Pulando...")
            continue

        print(f"\nProcessando subdivisão: {sup_split} -> {coco_split}")
        
        target_split_dir = os.path.join(output_dir, coco_split)
        os.makedirs(target_split_dir, exist_ok=True)

        # 1. Copia as imagens para a pasta de destino
        print("  Copiando arquivos de imagem...")
        for img_file in os.listdir(img_dir):
            src_img = os.path.join(img_dir, img_file)
            dst_img = os.path.join(target_split_dir, img_file)
            if os.path.isfile(src_img):
                shutil.copy2(src_img, dst_img)

        # 2. Converte as anotações do split e salva como JSON único
        print("  Convertendo anotações para o formato COCO...")
        out_json_path = os.path.join(target_split_dir, '_annotations.coco.json')
        convert_supervisely_to_coco(meta_path, ann_base_dir=ann_dir, save_as=out_json_path, only_img_name=True)
        print(f"  Finalizado! Arquivo JSON gerado em: {out_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Converte um dataset completo do formato Supervisely para a estrutura COCO.")
    parser.add_argument("input_dir", type=str, help="Caminho raiz do dataset Supervisely (deve conter meta.json, e pastas train, val).")
    parser.add_argument("output_dir", type=str, help="Caminho de saída para salvar o dataset formatado em COCO.")
    
    args = parser.parse_args()

    print(f'Lendo dataset Supervisely de: {args.input_dir}')
    print(f'Gerando estrutura COCO em: {args.output_dir}')
    
    process_full_dataset(args.input_dir, args.output_dir)
    print('\nConversão concluída com sucesso!')


if __name__ == "__main__":
    main()