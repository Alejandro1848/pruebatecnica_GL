from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

def scale_data(df, columns):
    """
    Escala los datos especificados de un DataFrame utilizando StandardScaler.
    Esto estandariza las características eliminando la media y escalando a la varianza unitaria.

    Parámetros:
    - df (DataFrame): El DataFrame de pandas desde el cual se extraen los datos.
    - columns (list): Lista de columnas a escalar.

    Retorna:
    - Array de numpy con los datos escalados.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df[columns])

def apply_kmeans(data_scaled, n_clusters=5, random_state=42):
    """
    Aplica el algoritmo K-Means a los datos escalados para realizar clustering.

    Parámetros:
    - data_scaled (array): Datos que han sido previamente escalados.
    - n_clusters (int): Número de clusters a formar.
    - random_state (int): Semilla para la generación de números aleatorios.

    Retorna:
    - Array de etiquetas indicando el cluster asignado a cada observación.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    return kmeans.fit_predict(data_scaled)

def visualize_clusters(data_scaled, labels):
    """
    Visualiza los resultados del clustering utilizando t-SNE para reducción de dimensionalidad.

    Parámetros:
    - data_scaled (array): Datos que han sido previamente escalados.
    - labels (array): Etiquetas de cluster asignadas por K-Means.

    Efecto secundario:
    - Muestra un gráfico de dispersión de los datos proyectados a dos dimensiones con t-SNE.
    """
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(data_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title('Visualización de Clusters con t-SNE')
    plt.xlabel('TSNE 1')
    plt.ylabel('TSNE 2')
    plt.show()

def count_clusters(labels):
    """
    Cuenta la frecuencia de cada cluster.

    Parámetros:
    - labels (array): Etiquetas de cluster asignadas por K-Means.

    Retorna:
    - Series de pandas con la cuenta de cada etiqueta de cluster.
    """
    return pd.Series(labels).value_counts()

def calculate_wcss(data_scaled, max_clusters=10):
    """
    Calcula la suma de las distancias cuadradas dentro de los clusters para diferentes valores de 'k'.

    Parámetros:
    - data_scaled (array): Datos escalados.
    - max_clusters (int): Número máximo de clusters para evaluar.

    Retorna:
    - Lista de WCSS para cada número de clusters desde 1 hasta max_clusters.
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_elbow(wcss):
    """
    Grafica la curva del método del codo para ayudar a determinar el número óptimo de clusters.

    Parámetros:
    - wcss (list): Lista de valores de WCSS.

    Efecto secundario:
    - Muestra un gráfico que facilita la elección del número de clusters basado en la curva de WCSS.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()




def calculate_silhouette(data_scaled, max_clusters=10):
    """
    Calcula los valores de silueta para diferentes números de clusters.

    Parámetros:
    - data_scaled (array): Datos que han sido previamente escalados.
    - max_clusters (int): Número máximo de clusters para evaluar.

    Retorna:
    - Diccionario con el número de clusters y el correspondiente valor de silueta.
    """
    silhouette_scores = {}
    for i in range(2, max_clusters + 1):  # El análisis de silueta no se puede calcular con un solo cluster
        kmeans = KMeans(n_clusters=i, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        silhouette_scores[i] = score
    return silhouette_scores

def plot_silhouette(silhouette_scores):
    """
    Grafica los valores de silueta para ayudar a determinar el número óptimo de clusters.

    Parámetros:
    - silhouette_scores (dict): Diccionario de valores de silueta indexados por el número de clusters.

    Efecto secundario:
    - Muestra un gráfico de los valores de silueta.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    plt.title('Silhouette Scores by Number of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()
