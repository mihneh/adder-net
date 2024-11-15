import sys
from pathlib import Path
import mlflow
import yaml
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import DatasetLoader, dummy_data_loader
from src.model import SimpleDenoisingModel
from src.evaluation import evaluate_model

def load_params(config_path='config/params.yaml'):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def main(model=None, data_loader=None):
    params = load_params()
    
    mlflow.start_run()

    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

    if model is None:
        model = SimpleDenoisingModel(params['input_dim'])
    
    if data_loader is None:
        data_loader = DatasetLoader(dummy_data_loader)

    evaluate_model(model, data_loader, sample_rate=params['sample_rate'])

    mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help="Path to the model")
    parser.add_argument('--data_loader', type=str, default=None, help="Path to the data loader")
    args = parser.parse_args()
    
    main(model=args.model, data_loader=args.data_loader)
