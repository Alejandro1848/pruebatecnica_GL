from google.colab import drive
import os
import pandas as pd

# Función para montar Google Drive
def mount_google_drive():
    drive.mount('/content/drive')

# Función para listar archivos en una carpeta específica
def list_files(folder_path):
    return os.listdir(folder_path)

# Función para cargar todos los dataframes desde una lista de archivos
def load_dataframes(folder_path, files):
    dataframes = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        if file.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.endswith('.xlsx'):
            df = pd.read_csv(file_path)
        dataframes[file] = df
    return dataframes
