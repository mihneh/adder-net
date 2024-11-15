import numpy as np
import mlflow
from pystoi import stoi

def compute_estoi(clean_signal: np.ndarray, denoised_signal: np.ndarray, sample_rate: int) -> float:
    return stoi(clean_signal, denoised_signal, sample_rate, extended=True)

def evaluate_model(model, data_loader, sample_rate: int = 16000):
    estoi_scores = []
    for idx, (noisy_audio, clean_audio) in enumerate(data_loader.load_data()):
        denoised_audio = model.predict(noisy_audio)
        estoi_value = compute_estoi(clean_audio, denoised_audio, sample_rate)
        estoi_scores.append(estoi_value)
        mlflow.log_metric(f'ESTOI_sample_{idx}', estoi_value)

    avg_estoi = np.mean(estoi_scores)
    mlflow.log_metric('Average_ESTOI', avg_estoi)
