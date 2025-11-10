import os
import yaml

def carregar_nomes_classes(dataset_name):
    yaml_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'yolo', 'yamls', dataset_name + '.yaml'))
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML não encontrado: {yaml_path}")
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']
