import os
import subprocess
import yaml
import optuna
import argparse
import re
from pathlib import Path

def extract_metric_from_log(log_lines):
    """
    Usa Regex para caçar o melhor resultado de Average Precision (IoU=0.50:0.95, area=all) 
    dentro do output padrão do SAM 3.
    Como ele imprime várias vezes (para BBox e Segm), pegamos o maior valor encontrado.
    """
    best_iou = 0.0
    
    # O padrão procura por: "Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.656"
    # Pegando especificamente o número do final
    pattern = re.compile(r"Average Precision\s*\(AP\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=100\s*\]\s*=\s*([0-9\.]+)")
    
    for line in log_lines:
        match = pattern.search(line)
        if match:
            try:
                val = float(match.group(1))
                if val > best_iou:
                    best_iou = val
            except:
                pass
                
    return best_iou if best_iou > 0 else None

def objective(trial, base_config_path, output_dir, epochs, gpus):
    lr_scale = trial.suggest_float("lr_scale", 0.01, 0.2, log=True)
    wd = trial.suggest_float("wd", 0.01, 0.2, log=True)
    focal_gamma = trial.suggest_categorical("focal_gamma", [1.5, 2.0, 2.5])
    
    trial_name = f"trial_{trial.number}"
    trial_dir = Path(output_dir) / trial_name
    
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
    
    hpo_configs_dir = Path("/workspace/sam3/train/configs/custom/hpo_trials")
    hpo_configs_dir.mkdir(parents=True, exist_ok=True)
    
    trial_yaml_name = f"{trial_name}.yaml"
    trial_yaml_path_absolute = hpo_configs_dir / trial_yaml_name
    
    with open(trial_yaml_path_absolute, 'w') as f:
        f.write("# @package _global_\n")
        yaml.dump(config, f)
        
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
        
    # Extrai o melhor resultado direto do output guardado na memória
    best_iou = extract_metric_from_log(full_log)
    
    if best_iou is None:
        print("⚠️ Aviso: O treinamento concluiu mas a métrica AP não foi encontrada no log.")
        raise optuna.TrialPruned("Métrica de validação não encontrada no log.")
        
    print(f"✅ Trial {trial.number} concluído. Melhor AP: {best_iou}")
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
    
    try:
        trial = study.best_trial
        print(f"  Valor (IoU / AP): {trial.value}")
        print("  Hiperparâmetros:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        best_hp_path = Path(args.project_dir) / "best_hyperparameters.yaml"
        with open(best_hp_path, 'w') as f:
            yaml.dump(trial.params, f)
        print(f"Salvo em {best_hp_path}")
    except ValueError as e:
        print("Nenhum Trial foi concluído com sucesso. O arquivo de hiperparâmetros não será gerado.")