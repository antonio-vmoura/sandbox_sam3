import argparse
import base64
import glob
import json
import os
import sys
import zlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as pymask


class NpEncoder(json.JSONEncoder):
    """
    Classe auxiliar para permitir que o serializador JSON converta
    corretamente os tipos numéricos gerados pela biblioteca NumPy.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_categories_from_meta(meta_json_path: str) -> dict:
    """
    Lê o arquivo meta.json e cria um dicionário mapeando o nome 
    da classe (ignora o background 'bg') para um ID numérico sequencial.
    """
    with open(meta_json_path, 'r', encoding='utf-8') as fs:
        json_meta = json.load(fs)
    
    classes = [clss['title'] for clss in json_meta['classes'] if clss['title'] != 'bg']
    mapCategories = {c: i for i, c in enumerate(classes)}
    return mapCategories


def get_all_ann_file(base_dir: str) -> tuple:
    """
    Busca todos os arquivos de anotação (.json) no diretório base.
    Retorna uma lista com os nomes dos arquivos e outra com o conteúdo JSON.
    """
    all_ann_files = glob.glob(os.path.join(base_dir, "*.json"))
    all_fname_img = [fname[:-5] for fname in all_ann_files]  # Remove a extensão '.json'
    all_json_ann = []
    
    for json_path in all_ann_files:
        with open(json_path, 'r', encoding='utf-8') as fs:
            json_suprv = json.load(fs)
        all_json_ann.append(json_suprv)
        
    return all_fname_img, all_json_ann


def decode_bitmap_mask(s: str) -> np.ndarray:
    """
    Decodifica a string em base64 (formato comprimido zlib do Supervisely)
    para um array binário do NumPy (máscara da imagem).
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)

    imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    
    if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
        mask = imdecoded[:, :, 3].astype(bool)  # Imagens de 4 canais
    elif len(imdecoded.shape) == 2:
        mask = imdecoded.astype(bool)  # Máscara 2D plana
    else:
        raise RuntimeError('Formato interno de máscara incorreto.')
        
    return mask


def convert_single_image(idimg: int, fname_img: str, json_suprv: dict, map_category: dict, imgs_base_dir: str, only_img_name: bool = False, start_annotation_id: int = 0) -> tuple:
    """
    Processa uma imagem individual, convertendo suas anotações geométricas 
    (bitmap ou polygon) para o formato padrão COCO (RLE, BBox e Area).
    """
    image_base = {
        "id": idimg,
        "width": json_suprv['size']['width'],
        "height": json_suprv['size']['height'],
        "file_name": Path(fname_img).name if only_img_name else fname_img,
        "license": 1,
        "date_captured": ""
    }

    # Ignora a classe de background
    objects = [obj for obj in json_suprv['objects'] if obj['classTitle'] != 'bg']

    h, w = json_suprv['size']['height'], json_suprv['size']['width']
    rles = []
    boxes = []
    areas = []
    
    for instance in objects:
        if instance['geometryType'] == 'bitmap':
            full_mask_img = np.zeros((h, w), dtype=bool)
            inst_mask = decode_bitmap_mask(instance['bitmap']['data'])
            inst_w, inst_h = inst_mask.shape
            
            # Posiciona a máscara nos limites originais da imagem
            id_h, id_w = instance['bitmap']['origin']
            full_mask_img[id_w:id_w+inst_w, id_h:id_h+inst_h] = inst_mask
            
            # Codifica a máscara booleana para RLE
            rle = pymask.encode(np.asfortranarray(full_mask_img)) 
            rle['counts'] = rle['counts'].decode('ascii')
            rles.append(rle)
            boxes.append(pymask.toBbox(rle))
            areas.append(pymask.area(rle))
            
        elif instance['geometryType'] == 'polygon':
            full_mask_img = np.zeros((h, w))
            interior_mask = np.zeros((h, w))
            pts = np.array(instance['points']['exterior']).astype(int)
            cv2.fillPoly(full_mask_img, [pts], 1)
            
            # Lida com polígonos internos (ex: buracos na lesão)
            if len(instance['points']['interior']) > 0:
                for inst_pts in instance['points']['interior']:
                    pts = np.array(inst_pts).astype(int)
                    cv2.fillPoly(interior_mask, [pts], 1) 
                    
            full_mask_img -= interior_mask
            
            # Codifica a máscara booleana para RLE
            rle = pymask.encode(np.asfortranarray(full_mask_img.astype(bool))) 
            rle['counts'] = rle['counts'].decode('ascii')
            rles.append(rle)
            boxes.append(pymask.toBbox(rle))
            areas.append(pymask.area(rle))

    # Cria a lista de anotações formatadas para o COCO
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


def convert_supervisely_to_coco(meta_path: str, ann_base_dir: str = './ds/ann/', save_as: str = None, only_img_name: bool = False) -> dict:
    """
    Função principal que orquestra a leitura da pasta de anotações e a 
    geração do dicionário estruturado no padrão COCO Dataset.
    """ 
    ann_fnames, ann_jsons = get_all_ann_file(ann_base_dir)
    map_category = get_categories_from_meta(meta_path)

    # Cria as representações de categorias do COCO
    catg_repr = [
        {
            "id": v,
            "name": k,
            "supercategory": "type"
        } 
        for k, v in map_category.items()
    ]

    out_cnv_imgs = [
        convert_single_image(id_img, ann_fnames[id_img], ann_jsons[id_img], map_category, ann_base_dir, only_img_name)
        for id_img in range(len(ann_fnames))
    ]
    
    # Separa imagens e anotações
    images_repr = [o[0] for o in out_cnv_imgs]
    ann_repr = [o[1] for o in out_cnv_imgs]

    # "Achata" a lista de anotações (de 2D para 1D)
    ann_repr_flatten = [inner for lst in ann_repr for inner in lst]

    # Ajusta os IDs das anotações sequencialmente
    for i, ann in enumerate(ann_repr_flatten):
        ann['id'] = i

    # Monta a estrutura final do JSON do COCO
    coco_fmt = {
        "info": {
            "year": datetime.now().strftime('%Y'),
            "version": "1.0",
            "description": "",
            "contributor": "converted from supervisely2coco - caiofcm",
            "url": "",
            "date_created": datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        },
        "images": images_repr,
        "annotations": ann_repr_flatten,
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "categories": catg_repr
    }

    # Salva o arquivo JSON final se o caminho for fornecido
    if save_as:
        with open(save_as, 'w', encoding='utf-8') as fp:
            json.dump(coco_fmt, fp, cls=NpEncoder)
            
    return coco_fmt


def main():
    """
    Configura e lida com os argumentos da linha de comando (CLI).
    """
    parser = argparse.ArgumentParser(description="""
    Supervisely2Coco:
    Converte anotações do formato Supervisely para o Formato COCO.
    Exemplo de uso via terminal:
        python supervisely_to_coco_converter.py meta.json './ds/ann/' formatted_coco.json
    """)
    parser.add_argument(
        "-v", "--version",
        help="Mostra a versão do script",
        action="version",
        version="Supervisely2Coco 0.0.1, Python {}".format(sys.version),
    )
    parser.add_argument("meta", type=str, help="Caminho para o arquivo Meta JSON")
    parser.add_argument("ann_base_dir", type=str, help="Diretório base das anotações (ex: './ds/ann/')")
    parser.add_argument("output", type=str, help="Caminho do arquivo JSON de saída (COCO)")
    parser.add_argument(
        '-n', '--only-image-name', 
        action='store_true', 
        help="Salva apenas o nome do arquivo da imagem (sem o caminho completo)"
    )    
    
    args = parser.parse_args()

    print(f'Convertendo a partir de meta={args.meta}; anotações em [{args.ann_base_dir}] para o arquivo {args.output}')
    
    # Chama a função de conversão com os parâmetros do CLI
    convert_supervisely_to_coco(
        meta_path=args.meta, 
        ann_base_dir=args.ann_base_dir, 
        save_as=args.output, 
        only_img_name=args.only_image_name
    )

    print('Conversão finalizada com sucesso.')


if __name__ == "__main__":
    main()