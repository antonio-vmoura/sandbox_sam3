import json
import os

def find_missing_images(json_path: str, img_dir: str) -> None:
    """
    Compara as imagens listadas no arquivo JSON com os arquivos reais do diretório.
    """
    if not os.path.exists(json_path):
        print(f"⚠️ Arquivo JSON não encontrado: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    missing_count = 0

    # Varre a lista de imagens do JSON e checa o disco
    for img in data.get('images', []):
        img_path = os.path.join(img_dir, img['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Imagem listada no JSON, mas ausente na pasta: {img['file_name']}")
            missing_count += 1

    # Relatório final do split
    if missing_count == 0:
        print(f"Tudo certo! Todas as imagens estão presentes na pasta.")
    else:
        print(f"Resumo: {missing_count} imagens faltando neste split.\n")


def main() -> None:
    """
    Ponto de entrada: define os caminhos e itera pelos splits do dataset.
    """
    base_dir = '/home/antoniovinicius/projects/sandbox_sam3/datasets/isic_2018_task1_coco'
    splits = ['train', 'valid', 'test']

    print("Iniciando auditoria de imagens do dataset...\n")

    for split in splits:
        print(f"--- Verificando divisão: {split.upper()} ---")
        
        json_path = os.path.join(base_dir, split, '_annotations.coco.json')
        img_dir = os.path.join(base_dir, split)
        
        find_missing_images(json_path, img_dir)
        print("-" * 40)


if __name__ == "__main__":
    main()
    
    
# Iniciando auditoria de imagens do dataset...

# --- Verificando divisão: TRAIN ---
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016062.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016072.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016051.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016057.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016056.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016063.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016070.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016058.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016064.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016068.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016066.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016055.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016061.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016059.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016060.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016048.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016054.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016050.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016065.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016069.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016071.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016052.jpg
# Imagem listada no JSON, mas ausente na pasta: ISIC_0016053.jpg
# Resumo: 23 imagens faltando neste split.

# ----------------------------------------
# --- Verificando divisão: VALID ---
# Tudo certo! Todas as imagens estão presentes na pasta.
# ----------------------------------------
# --- Verificando divisão: TEST ---
# Tudo certo! Todas as imagens estão presentes na pasta.
# ----------------------------------------