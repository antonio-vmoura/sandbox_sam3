import yaml
import argparse
from pathlib import Path

def create_phase3_yaml(base_yaml_path, best_hp_path, output_yaml_path, epochs, log_dir):
    # 1. Carrega o YAML da Fase 1 (Baseline)
    with open(base_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Carrega os melhores parâmetros achados pelo Optuna
    with open(best_hp_path, 'r') as f:
        best_hp = yaml.safe_load(f)

    # 3. Injeta os Hiperparâmetros
    print(f"-> Injetando hiperparâmetros otimizados: {best_hp}")
    config['scratch']['lr_scale'] = best_hp['lr_scale']
    config['scratch']['wd'] = best_hp['wd']
    
    if 'focal_gamma' in best_hp:
        try:
            for fn in config['roboflow_train']['loss']['loss_fns_find']:
                if 'IABCEMdetr' in fn['_target_']:
                    fn['gamma'] = best_hp['focal_gamma']
                if 'Masks' in fn['_target_']:
                    fn['focal_gamma'] = best_hp['focal_gamma']
        except KeyError:
            pass
            
    # 4. Ajusta parâmetros da Fase 3 (Épocas completas e nova pasta de logs)
    config['trainer']['max_epochs'] = epochs
    config['launcher']['experiment_log_dir'] = log_dir

    # 5. Salva o YAML da Fase 3 garantindo a tag do Hydra
    output_path = Path(output_yaml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# @package _global_\n") # OBRIGATÓRIO PARA O HYDRA NÃO QUEBRAR
        yaml.dump(config, f, sort_keys=False)
        
    print(f"-> YAML da Fase 3 gerado com sucesso em: {output_yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml", type=str, required=True)
    parser.add_argument("--hpo_yaml", type=str, required=True)
    parser.add_argument("--out_yaml", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()
    
    create_phase3_yaml(args.base_yaml, args.hpo_yaml, args.out_yaml, args.epochs, args.log_dir)