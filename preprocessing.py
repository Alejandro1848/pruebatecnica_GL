import pandas as pd

# Función para realizar la unión vertical de los DataFrames
def merge_dataframes(dfs):
    df_merged = pd.concat(dfs, ignore_index=True)
    return df_merged

#Función para verificar la integridad del proceso que combina los Df's
def verify_data_integrity(dfs, df_merged):
    expected_rows = sum([df.shape[0] for df in dfs])
    actual_rows = df_merged.shape[0]
    return expected_rows, actual_rows

# Función sencilla que elimina filas con valores nulos en una columna específica
def remove_missing_values(df, column_name):
    return df.dropna(subset=[column_name])

# Función que codifica datos categóricos basados en el promedio de una columna numérica
def encode_categorical_data(df, group_col, encode_col):

    mean_values = df.groupby(group_col)[encode_col].mean()
    df[f'{group_col} Encoded'] = df[group_col].map(mean_values)
    return df
