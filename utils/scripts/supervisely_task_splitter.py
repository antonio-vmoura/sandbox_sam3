import os
import json
import shutil

# Constantes definindo as classes alvo para cada tarefa
TASK1_CLASSES = ['skin cancer']
TASK2_CLASSES = ['globule', 'milia like cyst', 'negative network', 'pigment network', 'streaks']


def setup_directories(out_dir: str, splits: list):
    """
    Cria a estrutura de pastas base (ann e img) para um dataset de saída.
    """
    os.makedirs(out_dir, exist_ok=True)
    for split in splits:
        os.makedirs(os.path.join(out_dir, split, 'ann'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, split, 'img'), exist_ok=True)


def process_meta_file(meta_src: str, out_task1: str, out_task2: str):
    """
    Lê o arquivo meta.json original, filtra as classes e tags irrelevantes 
    para cada tarefa e salva as versões limpas nas pastas de destino.
    """
    if not os.path.exists(meta_src):
        print(f"Aviso: Arquivo {meta_src} não encontrado.")
        return

    with open(meta_src, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
        
    meta_t1 = meta_data.copy()
    meta_t2 = meta_data.copy()
    
    # Filtra as classes para manter apenas as correspondentes a cada tarefa
    meta_t1['classes'] = [c for c in meta_data.get('classes', []) if c['title'] in TASK1_CLASSES]
    meta_t2['classes'] = [c for c in meta_data.get('classes', []) if c['title'] in TASK2_CLASSES]
    
    # Remove tags indesejadas (evita poluir a interface do Roboflow)
    meta_t1['tags'] = [t for t in meta_data.get('tags', []) if 'task 2' not in t.get('name', '') and 'task 3' not in t.get('name', '')]
    meta_t2['tags'] = [t for t in meta_data.get('tags', []) if 'task 1' not in t.get('name', '') and 'task 3' not in t.get('name', '')]

    # Salva os novos arquivos meta.json
    with open(os.path.join(out_task1, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_t1, f, indent=4)
        
    with open(os.path.join(out_task2, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_t2, f, indent=4)


def process_split_files(src_dir: str, out_task1: str, out_task2: str, split: str):
    """
    Processa as anotações e imagens de uma divisão específica (treino, validação ou teste),
    alocando-as para a Tarefa 1, Tarefa 2 ou ambas, baseando-se nas tags.
    """
    split_dir = os.path.join(src_dir, split)
    if not os.path.isdir(split_dir):
        return
        
    ann_dir = os.path.join(split_dir, 'ann')
    img_dir = os.path.join(split_dir, 'img')

    # Ignora o processamento se as subpastas não existirem
    if not os.path.exists(ann_dir) or not os.path.exists(img_dir):
        return

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith('.json'):
            continue

        ann_path = os.path.join(ann_dir, ann_file)
        img_file = ann_file.replace('.json', '')
        img_path = os.path.join(img_dir, img_file)

        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        is_task1 = False
        is_task2 = False

        # Verifica as tags da imagem para identificar a qual tarefa ela pertence
        image_tags = data.get('tags', [])
        for t in image_tags:
            tag_name = t.get('name', '')
            if 'task 1: lesion segmentation' in tag_name:
                is_task1 = True
            elif 'task 2: attribution detection' in tag_name:
                is_task2 = True

        # Prepara e salva os dados referentes à Tarefa 1
        if is_task1:
            data_t1 = data.copy()
            data_t1['tags'] = [t for t in image_tags if 'task 2' not in t.get('name', '') and 'task 3' not in t.get('name', '')]
            data_t1['objects'] = [obj for obj in data.get('objects', []) if obj.get('classTitle') in TASK1_CLASSES]
            
            with open(os.path.join(out_task1, split, 'ann', ann_file), 'w', encoding='utf-8') as f:
                json.dump(data_t1, f, indent=4)
                
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(out_task1, split, 'img', img_file))

        # Prepara e salva os dados referentes à Tarefa 2
        if is_task2:
            data_t2 = data.copy()
            data_t2['tags'] = [t for t in image_tags if 'task 1' not in t.get('name', '') and 'task 3' not in t.get('name', '')]
            data_t2['objects'] = [obj for obj in data.get('objects', []) if obj.get('classTitle') in TASK2_CLASSES]
            
            with open(os.path.join(out_task2, split, 'ann', ann_file), 'w', encoding='utf-8') as f:
                json.dump(data_t2, f, indent=4)
                
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(out_task2, split, 'img', img_file))


def split_supervisely_dataset(src_dir: str, out_task1: str, out_task2: str):
    """
    Função orquestradora que divide o dataset original nos dois sub-datasets específicos.
    """
    splits = ['train', 'val', 'test']
    
    print("Criando estrutura de diretórios...")
    setup_directories(out_task1, splits)
    setup_directories(out_task2, splits)
    
    print("Processando e dividindo arquivo meta.json...")
    meta_src = os.path.join(src_dir, 'meta.json')
    process_meta_file(meta_src, out_task1, out_task2)

    print("Processando imagens e anotações por split...")
    for split in splits:
        process_split_files(src_dir, out_task1, out_task2, split)

    print("Processamento concluído com sucesso!")


def main():
    """
    Ponto de entrada do script com as definições de caminho.
    """
    original_path = '/home/antoniovinicius/projects/sandbox_sam3/datasets/isic_2018_all_tasks_supervisely'
    dataset_task1 = '/home/antoniovinicius/projects/sandbox_sam3/datasets/isic_2018_task1_supervisely'
    dataset_task2 = '/home/antoniovinicius/projects/sandbox_sam3/datasets/isic_2018_task2_supervisely'

    split_supervisely_dataset(original_path, dataset_task1, dataset_task2)

if __name__ == '__main__':
    main()