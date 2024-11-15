import os
import subprocess
import time
import socket

def start_mlflow_ui(address='127.0.0.1', start_port=5000, max_attempts=100):
    def find_free_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((address, port)) != 0

    port = start_port
    attempt = 0
    while not find_free_port(port) and attempt < max_attempts:
        attempt += 1
        port += 1

    if attempt == max_attempts:
        print(f"Не удалось найти свободный порт после {max_attempts} попыток.")
        return

    os.environ["MLFLOW_TRACKING_URI"] = "./mlruns_new"

    mlflow_command = f"mlflow ui --host {address} --port {port}"
    process = subprocess.Popen(mlflow_command, shell=True)

    time.sleep(2)

    print(f"MLflow UI is running at http://{address}:{port}")

if __name__ == "__main__":
    start_mlflow_ui(address='127.0.0.1', start_port=5000)
