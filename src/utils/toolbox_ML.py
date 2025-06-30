# En primer lugar hacemos todas las importaciones necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind, pearsonr

# Ahora, las diferentes funciones en orden:

def describe_df(df : pd.DataFrame):
    """
    Genera una tabla resumen con una descripción de las variables del DataFrame.

    Argumentos
    ----------
    `df` : pandas.DataFrame\\
        El DataFrame de entrada cuyas columnas vamos a analizar.

    Retorna
    -------
    pandas.DataFrame
        Una tabla resumen con los siguientes datos por fila:
        - DATA_TYPE: El tipo de dato (en python) de cada variable.
        - MISSINGS (%): El porcentaje de nulos/missings.
        - UNIQUE_VALUES: El número de valores únicos de cada columna.
        - CARDIN (%): La cardinalidad de cada variable.

    Notas
    -----
    El DataFrame de salida tiene por columnas las del DataFrame de entrada,
    y las métricas que acabamos de describir como filas.
    """

    # Fila 1: tipo de datos
    data_type = df.dtypes.astype(str)

    # Fila 2: porcentaje de missings
    missings = df.isnull().mean() * 100

    # Fila 3: número de valores únicos
    unique_values = df.nunique()

    # Fila 4: cardinalidad (en %)
    cardin = round((unique_values / len(df)) * 100, 2)

    # Creación del dataframe
    resumen = pd.DataFrame(
        [data_type, missings, unique_values, cardin],
        index=["DATA_TYPE", "MISSINGS (%)", "UNIQUE_VALUES", "CARDIN (%)"]
    )

    return resumen


def tipifica_variables(df : pd.DataFrame, umbral_categoria : int = 10, umbral_continua : float = 20):
    """
    Sugiere una clasificación para el tipo de cada variable basada en la cardinalidad y en el número de valores únicos.

    Argumentos
    ----------
    `df` : pandas.DataFrame\\
        El DataFrame de entrada cuyas variables vamos a clasificar.
    `umbral_categoria` : int, default=10\\
        Máximo número de valores únicos para considerar una variable como categórica.
    `umbral_continua` : float, default=20\\
        Porcentaje de cardinalidad a partir del cual consideramos una variable como numérica continua.

    Retorna
    -------
    pandas.DataFrame

    Un DataFrame con dos columnas:
    - `nombre_variable`: el nombre de cada variable del DataFrame original.
    - `tipo_sugerido`: el tipo de variable en el que se clasifica:
        * Binaria.
        * Categorica.
        * Numerica Discreta.
        * Numerica Continua.

    Notas
    -----
    - Si una variable tiene exactamente dos valores únicos, se clasifica como "Binaria".
    - Si el número de valores únicos es menor que `umbral_categoria`, se clasifica como "Categorica".
    - Si el numero de valores unicos es mayor o igual que `umbral_categoria`:
        * Si la cardinalidad es mayor o igual que `umbral_continua`, se clasifica como "Numerica Continua".
        * Si no (cardinalidad menor que `umbral_continua`), se clasifica como "Numerica Discreta".
    """

    tabla_tipificacion = []

    for col in df.columns: # Recorremos las columnas del df de entrada

        # Calculamos el número de valores únicos y la cardinalidad de cada variable
        valores_unicos = df[col].nunique()
        cardinalidad = (valores_unicos / len(df)) * 100

        # Clasificamos según los criterios del enunciado, usando los umbrales de entrada
        if valores_unicos == 2:
            tipo = "Binaria"
        elif valores_unicos < umbral_categoria:
            tipo = "Categorica"
        elif valores_unicos >= umbral_categoria:
            if cardinalidad >= umbral_continua:
                tipo = "Numerica Continua"
            else:
                tipo = "Numerica Discreta"

        # Añadimos la fila con la variable y su clasificación a lo que sera el DataFrame de salida
        tabla_tipificacion.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(tabla_tipificacion)


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Devuelve una lista de variables numéricas cuya correlación con la variable target es alta.

    Esta versión usa la matriz de correlación de pandas y no calcula el p-valor.
    Si se indica 'pvalue', se muestra un aviso de que no se está usando.
    
    Argumentos
    ----------
    `df` (pd.DataFrame):\\
        DataFrame con los datos.
    `target_col` (str):\\
        Columna objetivo (target) del modelo de regresión.
    `umbral_corr` (float):\\
        Valor entre 0 y 1 para el umbral de correlación.
    `pvalue` (float, opcional):\\
        No se usa en esta versión. Solo incluido para compatibilidad.

    Retorna:
    ----------
    lista or None: Lista con nombres de columnas numéricas que cumplen el umbral, o None si hay errores.
    """

    # Validaciones de entrada
    if not isinstance(target_col, str) or target_col not in df.columns:
        print("Error: 'target_col' debe ser una columna válida del DataFrame.")
        return None
    if not isinstance(umbral_corr, (float, int)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None
    if pvalue is not None:
        print("Aviso: Esta versión de la función no calcula p-valores. Se ignorará 'pvalue'.")

    # Comprobar que la columna target es numérica
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: La columna target debe ser numérica.")
        return None

    # Selección de columnas numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col not in numeric_cols:
        print("Error: La columna target no es numérica.")
        return None

    numeric_cols.remove(target_col)

    # Calculamos la correlación con la columna target
    correlaciones = df[numeric_cols + [target_col]].corr()[target_col]

    # Filtramos columnas con correlación absoluta mayor que el umbral
    columnas_seleccionadas = correlaciones[abs(correlaciones) >= umbral_corr].index.tolist()

    # Eliminamos la columna target si ha quedado dentro
    if target_col in columnas_seleccionadas:
        columnas_seleccionadas.remove(target_col)

    return columnas_seleccionadas


def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0.0, pvalue=None):
    """
    Genera pairplots de las columnas numéricas más correlacionadas con una variable objetivo.

    La función filtra las variables numéricas en base a un umbral de correlación y, opcionalmente,
    a un nivel de significación estadística. Si la lista de columnas es larga, divide la visualización
    en grupos de hasta 5 columnas (incluyendo siempre la variable target).
    
    Argumentos:
    ----------
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre de la variable objetivo (target).
    columns (list, opcional): Lista de nombres de columnas a considerar (por defecto, se usan todas las numéricas).
    umbral_corr (float): Umbral de correlación mínima (valor absoluto) para incluir una columna.
    pvalue (float, opcional): Nivel de significación estadística. Si se proporciona, se usa test de Pearson.

    Retorna:
    ----------
    lista or None: Lista de columnas que cumplen con los criterios o None si hay error.
    """

    # Validaciones de entrada
    
    if not isinstance(target_col, str) or target_col not in df.columns:
        print("Error: 'target_col' debe ser una columna existente del DataFrame.")
        return None
    if not isinstance(columns, list):
        print("Error: 'columns' debe ser una lista.")
        return None
    if not isinstance(umbral_corr, (float, int)) or not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None
    if pvalue is not None and (not isinstance(pvalue, float) or not (0 < pvalue < 1)):
        print("Error: 'pvalue' debe ser un float entre 0 y 1 o None.")
        return None
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: La columna 'target_col' debe ser numérica.")
        return None

    # Si columns está vacía, tomamos todas las variables numéricas excepto la target
    if not columns:
        columns = df.select_dtypes(include=np.number).columns.drop(target_col).tolist()
    else:
        # Filtramos columnas que existen en el DataFrame y son numéricas
        columns = [col for col in columns if col in df.columns and np.issubdtype(df[col].dtype, np.number)]

    columnas_filtradas = []

    for col in columns:
        try:
            correlacion = df[[target_col, col]].corr().iloc[0, 1]
            if abs(correlacion) >= umbral_corr:
                if pvalue is not None:
                    _, pval = pearsonr(df[target_col].dropna(), df[col].dropna())
                    if pval <= (1 - pvalue):
                        columnas_filtradas.append(col)
                else:
                    columnas_filtradas.append(col)
        except Exception as e:
            print(f"Advertencia: No se pudo calcular la correlación entre {target_col} y {col}. Error: {e}")

    if not columnas_filtradas:
        print("Ninguna columna cumple los criterios de correlación y significación.")
        return []

    # Dividir columnas en grupos de hasta 4 + target (máximo 5 por gráfico)
    max_columnas_por_plot = 4
    for i in range(0, len(columnas_filtradas), max_columnas_por_plot):
        grupo = columnas_filtradas[i:i + max_columnas_por_plot]
        sns.pairplot(df[[target_col] + grupo].dropna())
        plt.suptitle(f"Correlación con {target_col} (grupo {i // max_columnas_por_plot + 1})", y=1.02)
        plt.tight_layout()
        plt.show()

    return columnas_filtradas


def get_features_cat_regression(df, target_col, pvalue=0.05):
    """
    Identifica columnas categóricas significativamente asociadas con una variable numérica con alta cardinalidad.
    Esta función evalúa la relación entre variables categóricas y una variable objetivo mediante pruebas estadísticas:
    - Prueba t de Student si hay 2 categorías.
    - ANOVA de un factor si hay más de 2 categorías.

    Argumentos:
    ----------
    df (pandas.DataFrame): DataFrame con los datos.
    target_col (str): El nombre de la columna objetivo que se desea predecir.
    pvalue (float, opcional (default=0.05)): Umbral de significación estadística. Solo se seleccionan variables categóricas cuyo p-valor sea menor a este valor.

    Retorna: lista or None
    ----------
    Lista de nombres de columnas categóricas que muestran una diferencia estadísticamente significativa en el target_col. 
    None si hay errores o si no se encuentra ninguna.
    """ 

    # Comprobaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'df' debe ser un DataFrame.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns:
        print(f"Error: {target_col} debe ser una columna existente del DataFrame.")
        return None
    
    if not (0 < pvalue < 1):
        print("Error: 'pvalue' debe ser un float entre 0 y 1.")
        return None
    
    # Comprobar que es numerica con alta cardinalidad, siendo la proporción de valores únicos mayor al 5%
    num_unique = df[target_col].nunique()
    num_total = len(df[target_col].dropna())

    if df[target_col].dtype not in [np.int64, np.float64] or num_unique / num_total < 0.05:
        print(f"Error: La columna '{target_col}' no parece ser numérica con alta cardinalidad.")
        return None

    # Filtrar columnas categóricas
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if not cat_cols:
        print("No hay columnas categóricas en el DataFrame.")
        return None 

    lista_categoricas = []

    for col in cat_cols:
        if col == target_col:
            continue

        # Eliminar filas con nulos en col o target_col
        subset = df[[col, target_col]].dropna()

        # Agrupar los valores del target por categoría
        groups = subset.groupby(col)[target_col].apply(list)

        if len(groups) < 2:
            continue

        try:
            if len(groups) == 2: # t-test cuando es igual a dos
                stat, p = ttest_ind(*groups, equal_var=False)
            else: #ANOVA si hay más de dos grupos
                stat, p = f_oneway(*groups)

            if p < pvalue:
                lista_categoricas.append(col)
        
        except Exception as e:
            print("Error")
            continue

    return lista_categoricas


def plot_features_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Genera histogramas agrupados de la variable objetivo para analizar la relación con  otras variables. 
    Y devuelve la lista de columnas cuya relación con target_col es estadísticamente significativa.

    Argumentos:
    df (pd.DataFrame): DataFrame con los datos.
    target_col (str): Nombre de la variable objetivo (target).
    columns (list, opcional): Lista de nombres de columnas a considerar (por defecto, se usan todas las numéricas).
    pvalue (float, opcional): Nivel de significación estadística.
    with_individual_plot (bool, opcional (default=False)): Si es True, muestra los histogramas por variable significativa.

    Retorna: lista or None
    Lista de columnas cuya relación con target_col es estadísticamente significativa.
    None si hay errores o si no se encuentra ninguna.
    """
    
    # Comprobaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'df' debe ser un DataFrame.")
        return None

    if not isinstance(target_col, str) or target_col not in df.columns:
        print(f"Error: {target_col} debe ser una columna existente del DataFrame.")
        return None
    
    if not isinstance(columns, list):
        print("Error: 'columns' debe ser una lista.")
        return None
    
    if not (0 < pvalue < 1):
        print("Error: 'pvalue' debe ser un float entre 0 y 1.")
        return None
    
    # Comprobar que es numerica con alta cardinalidad, siendo la proporción de valores únicos mayor al 5%
    num_unique = df[target_col].nunique()
    num_total = len(df[target_col].dropna())

    if df[target_col].dtype not in [np.int64, np.float64] or num_unique / num_total < 0.05:
        print(f"Error: La columna '{target_col}' no parece ser numérica con alta cardinalidad.")
        return None

    # Si columns está vacía, usar las variables numéricas distintas de target_col
    if not columns:
        columns = [col for col in df.select_dtypes(include=np.number).columns if col != target_col]

    lista_columnas = []

    for col in columns:

        # Eliminar nulos
        subset = df[[col, target_col]].dropna()

        # Agrupar y test estadístico
        groups = subset.groupby(col)[target_col].apply(list)

        if len(groups) < 2:
            continue

        try:
            if len(groups) == 2: # t-test cuando es igual a dos
                stat, p = ttest_ind(*groups, equal_var=False)
            else: #ANOVA si hay más de dos grupos
                stat, p = f_oneway(*groups)

            if p < pvalue:
                lista_columnas.append(col)

        
            if with_individual_plot:

                if subset[col].dtype == "object": #con categóricas
                    plt.figure(figsize=(8, 5))
                    sns.histplot(data=subset, x=target_col, hue=col, alpha=0.5)
                    plt.title(f"Distribución de '{target_col}' según '{col}' (p={p:.4f})")
                    plt.xlabel(target_col)
                    plt.tight_layout()
                    plt.show()
                
                else: #con númericas
                    plt.figure(figsize=(8, 5))

                    if subset[col].nunique() / len(subset) > 0.05: #muchos valores únicos --> agrupamos
                        subset[f'{col}_bins'] = pd.cut(subset[col], 4)
                        sns.histplot(data=subset, x=target_col, hue=f'{col}_bins', alpha=0.5)
                    else:
                        sns.histplot(data=subset, x=target_col, hue=col, alpha=0.5)

                    plt.title(f"Distribución de '{target_col}' según '{col}' (p={p:.4f})")
                    plt.xlabel(target_col)
                    plt.tight_layout()
                    plt.show()
                    

        except Exception as e:
            print("Error")
            continue

    return lista_columnas