
# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Visualizaciones
# -----------------------------------------------------------------------
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# Analisis Exploratorio Series Temporales
# -----------------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Modelo Series Temporales
# -----------------------------------------------------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from itertools import product

# Otros
# -----------------------------------------------------------------------
from tqdm import tqdm

import os
import sys 

import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("src"))   
import soporte_preprocesamiento_serie_temporal as f



#EDA

def exploracion_dataframe(dataframe, columna_control, estadisticos = False):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    # Tipos de columnas
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    # Enseñar solo las columnas categoricas (o tipo objeto)
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene los siguientes valores únicos:")
        print(f"Mostrando {pd.DataFrame(dataframe[col].value_counts()).head().shape[0]} categorías con más valores del total de {len(pd.DataFrame(dataframe[col].value_counts()))} categorías ({pd.DataFrame(dataframe[col].value_counts()).head().shape[0]}/{len(pd.DataFrame(dataframe[col].value_counts()))})")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    if estadisticos == True:
        for categoria in dataframe[columna_control].unique():
            dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
            #Describe de objetos
            print("\n ..................... \n")

            print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe(include = "O").T)

            #Hacer un describe
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe().T)
    else: 
        pass
    print("\n----------\n")
    print("Las principales estadísticas de las variables númericas son:")
    display(dataframe.describe().T)

    print("\n----------\n")
    print("Las principales estadísticas de las variables categóricas son:")
    display(dataframe.describe(include = "O").T)

    print("\n----------\n")
    print("Las características principales del dataframe son:")
    display(dataframe.info())


##Comprobaciones 

class TimeSeriesAnalysis:
    def __init__(self, dataframe, temporal_column, value_column):
        """
        Inicializa el objeto TimeSeriesAnalysis.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            El DataFrame que contiene los datos de la serie temporal.
        temporal_column : str
            Nombre de la columna con las fechas o tiempo.
        value_column : str
            Nombre de la columna con los valores de la serie temporal.
        """
        self.data = dataframe.copy()
        self.temporal_column = temporal_column
        self.value_column = value_column

        # Asegurar que la columna temporal es de tipo datetime
        self.data[self.temporal_column] = pd.to_datetime(self.data[self.temporal_column])
        self.data.set_index(self.temporal_column, inplace=True)
    
    def comprobar_serie_continua(self):
        """
        Comprueba si la serie temporal es continua.
        """
        fecha_completa = pd.date_range(start=self.data.index.min(), end=self.data.index.max(), freq="MS")
        mes_anio_actual = self.data.index.to_period("M")
        mes_anio_completo = fecha_completa.to_period("M")
        meses_faltantes = mes_anio_completo.difference(mes_anio_actual)

        if len(meses_faltantes) == 0:
            print("La serie temporal es continua, no faltan meses.")
        else:
            print("La serie temporal NO es continua.")
            print("Meses-Años faltantes:", meses_faltantes)
    
    def graficar_serie(self):
        """
        Grafica la serie temporal original.
        """
        fig = px.line(
            self.data,
            x=self.data.index,
            y=self.value_column,
            title="Serie Temporal Original",
            labels={self.temporal_column: "Fecha", self.value_column: "Valores"}
        )
        fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Valores")
        fig.show()
    
    def graficar_media_movil(self, window=30):
        """
        Grafica la media móvil de la serie temporal.
        
        Parameters:
        -----------
        window : int
            Tamaño de la ventana para calcular la media móvil.
        """
        self.data["rolling_window"] = self.data[self.value_column].rolling(window=window).mean()
        fig = px.line(
            self.data,
            x=self.data.index,
            y=[self.value_column, "rolling_window"],
            title="Evolución con Media Móvil",
            labels={self.temporal_column: "Fecha", self.value_column: "Valores"}
        )
        fig.data[0].update(name="Valores Originales")
        fig.data[1].update(name=f"Media Móvil ({window} días)", line=dict(color="red"))
        fig.update_layout(template="plotly_white", xaxis_title="Fecha", yaxis_title="Valores")
        fig.show()
    
    def detectar_estacionalidad(self, figsize = (12, 10)):
        """
        Detecta visualmente si la serie temporal tiene un componente estacional.
        """
        decomposition = seasonal_decompose(self.data[self.value_column], model='additive', period=12)
        
        # Crear figura y subplots
        fig, axes = plt.subplots(4, 1, figsize= figsize, sharex=True)
        
        # Serie original
        axes[0].plot(self.data[self.value_column], color="blue", linewidth=2)
        axes[0].set_title("Serie Original", fontsize=14)
        axes[0].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Tendencia
        axes[1].plot(decomposition.trend, color="orange", linewidth=2)
        axes[1].set_title("Tendencia", fontsize=14)
        axes[1].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Estacionalidad
        axes[2].plot(decomposition.seasonal, color="green", linewidth=2)
        axes[2].set_title("Estacionalidad", fontsize=14)
        axes[2].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Ruido
        axes[3].plot(decomposition.resid, color="red", linewidth=2)
        axes[3].set_title("Ruido (Residuo)", fontsize=14)
        axes[3].grid(visible=True, linestyle="--", alpha=0.6)
        
        # Ajustar diseño
        plt.suptitle("Descomposición de la Serie Temporal", fontsize=16, y=0.95)
        plt.xlabel("Fecha", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def graficar_acf_pacf(self, lags=40):
        """
        Grafica las funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).
        
        Parameters:
        -----------
        lags : int
            Número de rezagos a graficar.
        """
        plt.figure(figsize=(12, 10))
        plot_acf(self.data[self.value_column].dropna(), lags=lags)
        plt.title("Función de Autocorrelación (ACF)")
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(12, 10))
        plot_pacf(self.data[self.value_column].dropna(), lags=lags, method="ywm")
        plt.title("Función de Autocorrelación Parcial (PACF)")
        plt.grid()
        plt.show()
    
    def prueba_estacionariedad(self):
        """
        Aplica la prueba de Dickey-Fuller aumentada para verificar estacionariedad.
        """
        result = adfuller(self.data[self.value_column].dropna())
        print("ADF Statistic:", result[0])
        print("p-value:", result[1])
        print("Valores Críticos:")
        for key, value in result[4].items():
            print(f"{key}: {value}")
        if result[1] < 0.05:
            print("Rechazamos la hipótesis nula. La serie es estacionaria.")
        else:
            print("No podemos rechazar la hipótesis nula. La serie NO es estacionaria.")

##Modelos

class SARIMAModel:
    def __init__(self):
        self.best_model = None
        self.best_params = None

    def generar_parametros(self, p_range, q_range, seasonal_order_ranges):
        """
        Genera combinaciones de parámetros SARIMA de forma automática.

        Args:
            p_range (range): Rango de valores para el parámetro p.
            q_range (range): Rango de valores para el parámetro q.
            seasonal_order_ranges (tuple of ranges): Rango de valores para los parámetros estacionales (P, D, Q, S).

        Returns:
            list of tuples: Lista con combinaciones en formato (p, q, (P, D, Q, S)).
        """
        P_range, D_range, Q_range, S_range = seasonal_order_ranges

        parametros = [
            (p, q, (P, D, Q, S))
            for p, q, (P, D, Q, S) in product(
                p_range, q_range, product(P_range, D_range, Q_range, S_range)
            )
        ]

        return parametros

    def evaluar_modelos(self, y_train, y_test, parametros, diferenciacion, df_length, variable):
        """
        Evalúa combinaciones de parámetros SARIMA, devuelve un DataFrame con los resultados,
        y genera una visualización de las predicciones comparadas con los valores reales.

        Args:
            y_train (pd.Series): Serie temporal de entrenamiento.
            y_test (pd.Series): Serie temporal de prueba.
            parametros (list of tuples): Lista de combinaciones de parámetros en formato [(p, q, (P, D, Q, S)), ...].
            diferenciacion (int): Valor para el parámetro `d` de diferenciación.
            df_length (int): Longitud total del dataset para calcular los índices de predicción.

        Returns:
            pd.DataFrame: DataFrame con las combinaciones de parámetros y los errores RMSE.
        """
        results = []

        for p, q, seasonal_order in tqdm(parametros):
            try:
                # Crear y entrenar el modelo SARIMAX
                modelo_sarima = SARIMAX(
                    y_train,
                    order=(p, diferenciacion, q),
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)


                start_index = len(y_train)
                end_index = df_length - 1
                pred_test = modelo_sarima.predict(start=start_index, end=end_index)
                pred_test = pd.Series(pred_test, index=y_test.index)  # Convertir a Serie de pandas

                # Calcular RMSE para el conjunto de prueba
                error = np.sqrt(mean_squared_error(y_test, pred_test))
                results.append({"p": p, "q": q, "seasonal_order": seasonal_order, "RMSE": error})

                # Guardar el mejor modelo
                if self.best_model is None or error < self.best_model["RMSE"]:
                    self.best_model = {
                        "modelo": modelo_sarima,
                        "RMSE": error,
                        "pred_test": pred_test,
                    }
                    self.best_params = {"p": p, "q": q, "seasonal_order": seasonal_order}

            except Exception as e:
                # Manejar errores durante el ajuste
                results.append({"p": p, "q": q, "seasonal_order": seasonal_order, "RMSE": None})

        # Convertir los resultados a un DataFrame
        results_df = pd.DataFrame(results)

        # Visualizar las predicciones del mejor modelo
        self._visualizar_predicciones_test(y_test, variable)
        return results_df


    def _visualizar_predicciones_test(self, y_test, variable):
        """
        Visualiza las predicciones del mejor modelo SARIMA comparando
        los valores reales y predicciones del conjunto de prueba, incluyendo
        el intervalo de confianza.

        Args:
            y_test (pd.Series): Serie temporal de prueba.
            variable (str): Nombre de la variable objetivo.
        """
        if self.best_model is None:
            raise ValueError("No se ha ajustado ningún modelo aún. Llama a 'evaluar_modelos' primero.")

        # Obtener las predicciones y el intervalo de confianza
        modelo = self.best_model["modelo"]
        forecast = modelo.get_forecast(steps=len(y_test))
        pred_test = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Crear la figura
        plt.figure(figsize=(14, 7))

        # Graficar valores reales
        sns.lineplot(x=y_test.index, y=y_test[variable], label="Valores Reales", color="blue", linewidth=2)

        # Graficar predicciones
        sns.lineplot(x=y_test.index, y=pred_test, label="Predicciones SARIMA", color="red", linestyle="--", linewidth=2)

        # Graficar intervalo de confianza
        plt.fill_between(
            y_test.index,
            conf_int.iloc[:, 0],  # Límite inferior
            conf_int.iloc[:, 1],  # Límite superior
            color="pink",
            alpha=0.3,
            label="Intervalo de Confianza",
        )

        # Personalización
        plt.title("Comparación de Predicciones vs Valores Reales (Conjunto de Prueba)", fontsize=16)
        plt.xlabel("Fecha", fontsize=14)
        plt.ylabel("Valores", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


























class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include=["O", "category"])
    
    def plot_numericas(self, color="grey", tamano_grafica=(20, 10)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20)
            axes[indice].set_title(f"Distribución de {columna}")
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout()

        if len(lista_num) % 2 !=0:
            fig.delaxes(axes[-1])


    def plot_categoricas(self, color="grey", tamano_grafica=(20, 10)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_cat = self.separar_dataframes()[1].columns
        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_cat) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(lista_cat):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.suptitle("Distribución de variables categóricas")
        plt.tight_layout()

        if len(lista_cat) % 2 !=0:
            fig.delaxes(axes[-1])


    def plot_relacion(self, vr, tamano_grafica=(20, 10), tamanio_fuente=18):
        """
        Genera gráficos que muestran la relación entre cada columna del DataFrame y una variable de referencia (vr).
        Los gráficos son adaptativos según el tipo de dato: histogramas para variables numéricas y countplots para categóricas.

        Parámetros:
        -----------
        vr : str
            Nombre de la columna que actúa como la variable de referencia para las relaciones.
        tamano_grafica : tuple, opcional
            Tamaño de la figura en el formato (ancho, alto). Por defecto es (20, 10).
        tamanio_fuente : int, opcional
            Tamaño de la fuente para los títulos de los gráficos. Por defecto es 18.

        Retorno:
        --------
        None
            Muestra una serie de subgráficos con las relaciones entre la variable de referencia y el resto de columnas del DataFrame.

        Notas:
        ------
        - La función asume que el DataFrame de interés está definido dentro de la clase como `self.dataframe`.
        - Se utiliza `self.separar_dataframes()` para obtener las columnas numéricas y categóricas en listas separadas.
        - La variable de referencia (`vr`) no será graficada contra sí misma.
        - Los gráficos utilizan la paleta "magma" para la diferenciación de categorías o valores de la variable de referencia.
        """

        lista_num = self.separar_dataframes()[0].columns
        lista_cat = self.separar_dataframes()[1].columns

        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(self.dataframe.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in lista_num:
                sns.histplot(x = columna, 
                             hue = vr, 
                             data = self.dataframe, 
                             ax = axes[indice], 
                             palette = "magma", 
                             legend = False)
                
            elif columna in lista_cat:
                sns.countplot(x = columna, 
                              hue = vr, 
                              data = self.dataframe, 
                              ax = axes[indice], 
                              palette = "magma"
                              )

            axes[indice].set_title(f"Relación {columna} vs {vr}",size=tamanio_fuente)   

        plt.tight_layout()
    
        
    def deteccion_outliers(self, color = "grey"):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(ncols = 2, nrows = math.ceil(len(lista_num)/2), figsize=(20,10))
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})
            axes[indice].set_title(f"Outliers {columna}")  

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada 

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="magma",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)

##ENCODING
def codificar(dataframe):
    """
    Codifica las columnas categóricas del DataFrame.

    Este método reemplaza los valores de las columnas categóricas por sus frecuencias relativas dentro de cada
    columna.

    Returns:
        - pd.DataFrame. El DataFrame con las columnas categóricas codificadas.
    """
    # Sacamos el nombre de las columnas categóricas
    col_categoricas = dataframe.select_dtypes(include=["category", "object"]).columns

    # Iteramos por cada una de las columnas categóricas para aplicar el encoding
    for categoria in col_categoricas:
        # Calculamos las frecuencias de cada una de las categorías
        frecuencia = dataframe[categoria].value_counts(normalize=True)

        # Mapeamos los valores obtenidos en el paso anterior, sobreescribiendo la columna original
        dataframe[categoria] = dataframe[categoria].map(frecuencia)

    return dataframe

