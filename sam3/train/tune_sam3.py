import os
import subprocess
import yaml
import optuna
import argparse
import pandas as pd
from pathlib import Path
import json

def get_best_metric_from_log(experiment_dir):
    """
    Lê os resultados gerados pelo coco_evaluator_offline.
    O SAM3 salva as métricas no diretório de dumps.
    """
    dump_dir = Path(experiment_dir) / "dumps"
    
    # Procura os arquivos de métricas (geralmente json ou txt com os resultados do COCO)
    # Como o SAM3 salva as métricas finais, vamos buscar o maior AP de segmentação (IoU)
    best_iou = 0.0
    
    # O COCO evaluator salva os resultados padrão em um arquivo. 
    # Aqui vamos tentar ler os resultados caso ele exporte um JSON, 
    # ou analisar a saída padrão se não houver um arquivo fácil.
    # Assumindo que você salva métricas em tensorboard/logs, vamos tentar achar o arquivo summary
    # Se o trainer salvar um 'val_stats.json' ou 'metrics.json':
    log_files = list(Path(experiment_dir).rglob("*.json"))
    for file in log_files:
        if "coco" in file.name.lower() or "eval" in file.name.lower():
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    # Busca a métrica AP 50-95 ou AP 50 para máscaras
                    # O nome exato depende do logger do SAM3, geralmente "segm_AP"
                    for k, v in data.items():
                        if 'segm' in k.lower() and 'ap' in k.lower() and isinstance(v, (int, float)):
                            if v > best_iou:
                                best_iou = v
            except:
                continue
                
    return best_iou if best_iou > 0 else None

def objective(trial, base_config_path, output_dir, epochs, gpus):
    # 1. Sugerir Hiperparâmetros
    lr_scale = trial.suggest_float("lr_scale", 0.01, 0.2, log=True)
    wd = trial.suggest_float("wd", 0.01, 0.2, log=True)
    focal_gamma = trial.suggest_categorical("focal_gamma", [1.5, 2.0, 2.5])
    
    trial_name = f"trial_{trial.number}"
    trial_dir = Path(output_dir) / trial_name
    
    # 2. Carregar o YAML base e aplicar as modificações
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['scratch']['lr_scale'] = lr_scale
    config['scratch']['wd'] = wd
    
    try:
        loss_fns = config['roboflow_train']['loss']['loss_fns_find']
        for fn in loss_fns:
            if 'IABCEMdetr' in fn['_target_']:
                fn['gamma'] = focal_gamma
            if 'Masks' in fn['_target_']:
                fn['focal_gamma'] = focal_gamma
    except KeyError:
        pass
    
    config['trainer']['max_epochs'] = epochs
    config['trainer']['val_epoch_freq'] = max(1, epochs // 5)
    config['launcher']['experiment_log_dir'] = str(trial_dir)
    config['launcher']['gpus_per_node'] = gpus
    
    # 3. Salvar o YAML temporário na pasta "sam3/train/configs/custom/hpo_trials"
    hpo_configs_dir = Path("/workspace/sam3/train/configs/custom/hpo_trials")
    hpo_configs_dir.mkdir(parents=True, exist_ok=True)
    
    trial_yaml_name = f"{trial_name}.yaml"
    trial_yaml_path_absolute = hpo_configs_dir / trial_yaml_name
    
    # Gravando com a diretiva obrigatória do Hydra
    with open(trial_yaml_path_absolute, 'w') as f:
        f.write("# @package _global_\n")
        yaml.dump(config, f)
        
    # 4. Executar o Treinamento do SAM 3 com o caminho relativo
    cmd = [
        "python", "sam3/train/train.py",
        "-c", f"configs/custom/hpo_trials/{trial_yaml_name}",
        "--use-cluster", "0"
    ]
    
    print(f"\n--- Iniciando {trial_name} com lr_scale={lr_scale:.4f}, wd={wd:.4f} ---")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    full_log = []
    for line in process.stdout:
        full_log.append(line)
        if "Epoch:" in line or "loss" in line.lower() or "ap" in line.lower():
            print(line.strip())
            
    process.wait()
    
    if process.returncode != 0:
        print(f"❌ ERRO CRÍTICO NO TRIAL {trial.number}. Últimas linhas do log:")
        print("".join(full_log[-20:]))
        raise optuna.TrialPruned("O treinamento falhou ou gerou erro (ex: OOM).")
        
    # 5. Ler o resultado
    best_iou = get_best_metric_from_log(trial_dir)
    
    if best_iou is None:
        raise optuna.TrialPruned("Métrica de validação não encontrada no log.")
        
    return best_iou

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml", type=str, required=True, help="YAML de configuracao base")
    parser.add_argument("--project_dir", type=str, required=True, help="Diretorio de saida do HPO")
    parser.add_argument("--trials", type=int, default=15, help="Numero de tentativas do Optuna")
    parser.add_argument("--epochs", type=int, default=10, help="Épocas por trial (manter baixo)")
    parser.add_argument("--gpus", type=int, default=1, help="Numero de GPUs a usar")
    args = parser.parse_args()
    
    study_name = "sam3_hpo"
    storage_name = f"sqlite:///{args.project_dir}/{study_name}.db"
    
    os.makedirs(args.project_dir, exist_ok=True)
    
    # Maximizar o Jaccard Index (IoU) / Mask AP
    study = optuna.create_study(
        study_name=study_name, 
        direction="maximize", 
        storage=storage_name, 
        load_if_exists=True
    )
    
    study.optimize(
        lambda trial: objective(trial, args.base_yaml, args.project_dir, args.epochs, args.gpus), 
        n_trials=args.trials
    )
    
    print("\n=======================================================")
    print("HPO Concluído!")
    print("Melhor Trial:")
    trial = study.best_trial
    print(f"  Valor (IoU / AP): {trial.value}")
    print("  Hiperparâmetros:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Salva os melhores hiperparâmetros num arquivo final
    best_hp_path = Path(args.project_dir) / "best_hyperparameters.yaml"
    with open(best_hp_path, 'w') as f:
        yaml.dump(trial.params, f)
    print(f"Salvo em {best_hp_path}")