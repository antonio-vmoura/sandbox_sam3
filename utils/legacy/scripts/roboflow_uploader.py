import subprocess
import sys


def upload_to_roboflow(workspace: str, project_name: str, dataset_path: str) -> None:
    """
    Constrói e executa o comando CLI do Roboflow para importar o dataset
    para o workspace e projeto especificados.
    """
    print("Preparando para importar o dataset da Tarefa 1...")
    print(f"Caminho: {dataset_path}")
    print(f"Destino: Workspace '{workspace}', Projeto '{project_name}'")
    
    # Monta o comando como uma lista de argumentos (mais seguro que uma string única)
    command = [
        "roboflow", 
        "import", 
        "-w", workspace, 
        "-p", project_name, 
        dataset_path
    ]
    
    try:
        # Executa o comando no terminal e aguarda a finalização (check=True levanta erro se falhar)
        subprocess.run(command, check=True)
        
        print("\nUpload concluído com sucesso!")
        print("Verifique a aba 'Annotate' no Roboflow para confirmar se as imagens e os polígonos foram carregados corretamente.")
        
    except subprocess.CalledProcessError:
        # Captura erros caso o comando retorne um status de falha (ex: projeto não encontrado)
        print("\nHouve um erro durante o upload.")
        print("Verifique se você está autenticado (rodou 'roboflow login') e se o nome do workspace/projeto estão corretos.")
        sys.exit(1)
        
    except FileNotFoundError:
        # Captura erro caso a CLI do Roboflow não esteja instalada no ambiente atual
        print("\nErro: O comando 'roboflow' não foi encontrado no terminal.")
        print("Certifique-se de ter instalado a biblioteca rodando: pip install roboflow")
        sys.exit(1)


def main() -> None:
    """
    Função principal que define as variáveis de ambiente e chama a função de upload.
    """
    # Certifique-se de ter rodado `roboflow login` no terminal antes de executar
    workspace_name = "vmouramaster"
    project_id = "isic_2018"
    dataset_directory = "/home/antoniovinicius/projects/sandbox_sam3/datasets/isic_task1_segmentation"

    upload_to_roboflow(workspace_name, project_id, dataset_directory)


# Garante que o script só execute as funções se for chamado diretamente
if __name__ == "__main__":
    main()