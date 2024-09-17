# README - Modelo de Clusterización para el Purchase Card Transactions.


Este proyecto consiste en una serie de scripts y un notebook principal que juntos forman una pipeline para la carga, preprocesamiento, análisis, implementación de un modelo de clusterización y la visualización de datos de los resultados que nos arroja el modelo. Específicamente enfocada en el clustering usando K-Means y la evaluación de la calidad del clustering con el método del codo y análisis de silueta. El notebook principal controla el flujo del proceso y hace uso de funciones contenidas en los módulos importados para ejecutar tareas específicas.

## Estructura del Proyecto
El proyecto está organizado en los siguientes archivos:

data_loader.py: Contiene funciones para cargar datos desde Google Drive y listar archivos en un directorio especificado.
preprocessing.py: Proporciona funciones para manipular y preparar los datos antes del análisis, como la combinación de DataFrames y la limpieza de datos.
training.py: Incluye funciones para escalar datos, aplicar el algoritmo K-Means, realizar el análisis del codo y silueta, y visualizar los resultados del clustering.
Purchases.ipynb: Es el notebook principal que utiliza los módulos mencionados para ejecutar el flujo completo desde la carga de datos hasta la visualización de los resultados del clustering.

## Descripción de Funciones
### data_loader.py

mount_google_drive(): Monta Google Drive para acceder a los archivos almacenados.
list_files(folder_path): Devuelve una lista de archivos en el directorio especificado.
load_dataframes(folder_path, files): Carga DataFrames desde los archivos listados en el directorio.

### preprocessing.py

merge_dataframes(dfs): Combina múltiples DataFrames en uno solo.
verify_data_integrity(dfs, df_merged): Verifica la integridad de los datos después de la combinación.
remove_missing_values(df, column_name): Elimina filas con valores nulos en columnas especificadas.
encode_categorical_data(df, group_col, encode_col): Codifica datos categóricos basados en el promedio de una columna numérica.

### model_app.py

scale_data(df, columns): Escala datos numéricos usando StandardScaler.
apply_kmeans(data_scaled, n_clusters, random_state): Aplica el algoritmo K-Means a los datos escalados.
visualize_clusters(data_scaled, labels): Visualiza los clusters usando t-SNE.
count_clusters(labels): Cuenta la frecuencia de cada cluster.
calculate_wcss(data_scaled, max_clusters): Calcula la suma de las distancias cuadradas dentro de los clusters para diferentes valores de 'k'.
plot_elbow(wcss): Grafica la curva del método del codo.
calculate_silhouette(data_scaled, max_clusters): Calcula los valores de silueta para diferentes números de clusters.
plot_silhouette(silhouette_scores): Grafica los valores de silueta.

### Flujo del Notebook Principal (Purchases.ipynb)
El notebook Purchases.ipynb sigue los siguientes pasos en su ejecución:

Montar Google Drive para acceder a los datos.
Listar y cargar los datos desde una ubicación especificada en Google Drive.
Preprocesar los datos combinando, limpiando, y codificando según sea necesario.
Escalar los datos preparatorios para el análisis de clustering.
Evaluar el número óptimo de clusters usando el método del codo y el análisis de silueta.
Aplicar K-Means con el número óptimo de clusters determinado.
Visualizar los resultados del clustering y contar la distribución de los clusters.

Instrucciones de Uso
Para utilizar este proyecto, siga los pasos:

Asegúrese de tener acceso a Google Colab o un entorno similar que pueda montar Google Drive.
Coloque los scripts en la misma carpeta que el notebook o asegúrese de que las rutas de importación sean correctas.
Ejecute el notebook Purchases.ipynb desde el principio hasta el final para ver el flujo completo de análisis y visualización.
