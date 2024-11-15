import numpy as np

class DatasetLoader:
    def __init__(self, data_loader_func):
        self.data_loader_func = data_loader_func

    def load_data(self):
        return self.data_loader_func()
        
def dummy_data_loader():
    clean_signal = np.random.randn(16000)
    noisy_signal = clean_signal + np.random.randn(16000) * 0.1
    return [(noisy_signal, clean_signal)]
