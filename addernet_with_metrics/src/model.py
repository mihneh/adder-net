import torch
import torch.nn as nn
import numpy as np

class SimpleDenoisingModel(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleDenoisingModel, self).__init__()
        self.denoise_layer = nn.Linear(input_dim, input_dim)
    
    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        return self.denoise_layer(noisy_audio)
    
    def predict(self, noisy_audio: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            noisy_tensor = torch.tensor(noisy_audio, dtype=torch.float32)
            denoised_tensor = self(noisy_tensor)
            return denoised_tensor.numpy()
