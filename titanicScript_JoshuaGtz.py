import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import Counter 

# --- Carga y Preparación de Datos (Data Wrangling) ---
Ruta_Datos = r'D:\UAQ\UAQ Semestres\10mo Semestre\Algoritmos de IA\1er EP - Titanic\train.csv'
Registro_Pasajeros = pd.read_csv(Ruta_Datos, sep=',')
# Renombrar columnas para consistencia
Registro_Pasajeros.rename(columns={'Survived': 'Sobrevivio', 'Sex': 'Genero'}, inplace=True)
print("--- Dimensión del Conjunto de Datos ---")
print(f"Filas: {Registro_Pasajeros.shape[0]}, Columnas: {Registro_Pasajeros.shape[1]}")

# Gestión de Datos Faltantes (Imputación)
Registro_Pasajeros.loc[Registro_Pasajeros['Age'].isnull(), 'Age'] = Registro_Pasajeros['Age'].mean() # Imputar Edad con la media
Puerto_Moda = Registro_Pasajeros['Embarked'].value_counts().index[0]
Registro_Pasajeros['Embarked'].fillna(Puerto_Moda, inplace=True) # Imputar Puerto con la moda
Registro_Pasajeros['Fare'].fillna(Registro_Pasajeros['Fare'].median(), inplace=True) # Imputar Tarifa con la mediana

# Codificación Categórica (One-Hot Encoding)
Datos_Codificados = pd.get_dummies(Registro_Pasajeros, columns=['Genero', 'Embarked'], prefix=['Es', 'Puerto'], drop_first=True)

# Definición de Atributos (X) y Objetivo (Y)
Atributos_Seleccionados = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Es_male', 'Puerto_Q', 'Puerto_S']
X_datos = Datos_Codificados[Atributos_Seleccionados].values
Y_objetivo = Datos_Codificados['Sobrevivio'].values

# --- Implementación de División de Datos (Split) ---
def Generar_Division_Entrenamiento_Prueba(X, Y, Fraccion_Prueba=0.2, Semilla_Aleatoria=42):
    if Semilla_Aleatoria is not None:
        np.random.seed(Semilla_Aleatoria)
    Tamanio_Total = X.shape[0]
    Cantidad_Prueba = int(Tamanio_Total * Fraccion_Prueba)
    Indices = np.arange(Tamanio_Total)
    np.random.shuffle(Indices)
    # División basada en índices aleatorios
    Indices_Prueba = Indices[:Cantidad_Prueba]
    Indices_Entrenamiento = Indices[Cantidad_Prueba:]
    X_entrenamiento, X_prueba = X[Indices_Entrenamiento], X[Indices_Prueba]
    Y_entrenamiento, Y_prueba = Y[Indices_Entrenamiento], Y[Indices_Prueba]
    return X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba

X_Ent, X_Prueba, Y_Ent, Y_Prueba = Generar_Division_Entrenamiento_Prueba(X_datos, Y_objetivo, Fraccion_Prueba=0.2, Semilla_Aleatoria=42)

# --- Implementación de Distancia y Clasificador k-NN ---
def Calculo_Distancia_Euclidiana(Punto1, Punto2):
    return np.linalg.norm(Punto1 - Punto2) # Distancia Euclideana usando norma L2

class Clasificador_KNN_Manual:
    def __init__(self, K=5):
        self.K = K 
        self.X_base = None
        self.Y_base = None

    def Ajustar(self, X_base, Y_base):
        self.X_base = X_base # Almacena datos de entrenamiento
        self.Y_base = Y_base

    def _Determinar_Clase_Unica(self, Punto_Evaluar):
        Lista_Distancias = []
        # 1. Calcular distancia a todos los vecinos
        for X_vecino, Y_etiqueta in zip(self.X_base, self.Y_base):
            dist = Calculo_Distancia_Euclidiana(Punto_Evaluar, X_vecino)
            Lista_Distancias.append((dist, Y_etiqueta))
 
        # 2. Obtener los K vecinos más cercanos
        Lista_Distancias.sort(key=lambda item: item[0])
        Etiquetas_Vecinos = [item[1] for item in Lista_Distancias[:self.K]]

        # 3. Voto por mayoría
        Voto_Final = Counter(Etiquetas_Vecinos).most_common(1)[0][0]
        return Voto_Final

    def Predecir(self, X_Prueba):
        return np.array([self._Determinar_Clase_Unica(x) for x in X_Prueba])

    def Calcular_Metrica(self, X_Prueba, Y_Prueba):
        Predichos = self.Predecir(X_Prueba)
        return np.mean(Predichos == Y_Prueba) # Calcula la precisión (accuracy)
 
# ----------------------------------------------------------------------------------
# --- Ejecución del Modelo y Optimización de K (Método del Codo) ---
# ----------------------------------------------------------------------------------

K_Min_Optimizacion = 7 
K_Max_Optimizacion = 15
Rango_K_Especial = range(K_Min_Optimizacion, K_Max_Optimizacion + 1)
Rango_K_Grafico = range(1, 15)

def Metodo_Codo_Personalizado_Optimizacion(X_ent, Y_ent, X_pr, Y_pr, Rango_K):
    Tasa_Errores = []
    Precisiones = []
    for k in Rango_K:
        modelo_k = Clasificador_KNN_Manual(K=k)
        modelo_k.Ajustar(X_ent, Y_ent)
        precision_k = modelo_k.Calcular_Metrica(X_pr, Y_pr)
        Precisiones.append(precision_k)
        Tasa_Errores.append(1 - precision_k)
    return list(Rango_K), Tasa_Errores, Precisiones

# 1. Ejecutar optimización para el rango 7 a 15
K_Valores_Optimizacion, K_Errores_Optimizacion, K_Precisiones = Metodo_Codo_Personalizado_Optimizacion(
    X_Ent, Y_Ent, X_Prueba, Y_Prueba, Rango_K_Especial)

# 2. Determinar el K óptimo (máxima precisión)
indice_mejor_k = np.argmax(K_Precisiones)
K_Optimo = K_Valores_Optimizacion[indice_mejor_k]
Precision_Optima = K_Precisiones[indice_mejor_k]

# 3. Calcular errores para el rango completo para la gráfica (1 a 14)
K_Valores_Grafico, K_Errores_Grafico, _ = Metodo_Codo_Personalizado_Optimizacion(
    X_Ent, Y_Ent, X_Prueba, Y_Prueba, Rango_K_Grafico)

# 4. Establecer el K óptimo como el valor base
Valor_K_Base = K_Optimo 

print(f"\n--- Determinación del K Óptimo (Rango K={K_Min_Optimizacion} a K={K_Max_Optimizacion}) ---")
print(f"K Óptimo Encontrado: {K_Optimo} (Precisión: {Precision_Optima:.4f})")
print(f"K Base para Métricas y Gráficas establecida en: {Valor_K_Base}")

# 5. Entrenar modelo final con K Óptimo
Modelo_Base_KNN = Clasificador_KNN_Manual(K=Valor_K_Base)
Modelo_Base_KNN.Ajustar(X_Ent, Y_Ent)
Y_Pred_Base = Modelo_Base_KNN.Predecir(X_Prueba)


# --- Cálculo de Métricas Detalladas (Confusion Matrix & Scores) ---

def Calcular_Metricas_Clasificacion(Y_Verdadero, Y_Predicho, Etiqueta_Positiva=1):
    # Cálculo de los 4 componentes de la Matriz de Confusión
    TP = np.sum((Y_Verdadero == Etiqueta_Positiva) & (Y_Predicho == Etiqueta_Positiva))
    TN = np.sum((Y_Verdadero != Etiqueta_Positiva) & (Y_Predicho != Etiqueta_Positiva))
    FP = np.sum((Y_Verdadero != Etiqueta_Positiva) & (Y_Predicho == Etiqueta_Positiva))
    FN = np.sum((Y_Verdadero == Etiqueta_Positiva) & (Y_Predicho != Etiqueta_Positiva))

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

    Matriz_Confusion = np.array([[TN, FP], [FN, TP]]) # Formato: [[TN, FP], [FN, TP]]
    return Matriz_Confusion, Precision, Recall, F1

Matriz, Prec, Rec, F1_Score = Calcular_Metricas_Clasificacion(Y_Prueba, Y_Pred_Base, Etiqueta_Positiva=1)

print(f"\n--- Métricas del Modelo (K={Valor_K_Base}) ---")
print(f"Matriz de Confusión (TN, FP / FN, TP):\n {Matriz}")
print(f"Precisión: {Prec:.4f}")
print(f"Recall (Sensibilidad): {Rec:.4f}")
print(f"F1-Score: {F1_Score:.4f}")


# ----------------------------------------------------------------------------------
# --- Sección de Visualizaciones Finales ---
# ----------------------------------------------------------------------------------
print("\n--- Visualizaciones Generadas ---")

## 1. Gráfico del Método del Codo (Elbow Plot) con Curva Cuadrática
plt.figure(figsize=(9, 6))
# 1.1 Datos Originales
plt.plot(K_Valores_Grafico, K_Errores_Grafico, marker='s', linestyle='--', color='darkred', linewidth=1, alpha=0.6, label='Tasa de Error (Datos Reales)')

# 1.2 Regresión Cuadrática (Grado 2)
K_Valores_np = np.array(K_Valores_Grafico)
coeficientes_polinomio = np.polyfit(K_Valores_np, K_Errores_Grafico, 2) # GRADO 2 (Cuadrática)
funcion_polinomio = np.poly1d(coeficientes_polinomio)
curva_suavizada = funcion_polinomio(K_Valores_np)

# 1.3 Curva Suavizada y K Óptimo
plt.plot(K_Valores_np, curva_suavizada, linestyle='-', color='dodgerblue', linewidth=3, label='Regresión Cuadrática (Grado 2)', alpha=0.8)
plt.axvline(x=Valor_K_Base, color='red', linestyle='-', linewidth=2, label=f'K Óptimo Seleccionado ({Valor_K_Base})')
plt.text(Valor_K_Base + 0.5, min(K_Errores_Grafico) + 0.005, f'K Óptimo ({Valor_K_Base})', color='red', weight='bold')
plt.title('1. Método del Codo para k-NN: Tasa de Error vs. K (con Regresión Cuadrática)', fontsize=14)
plt.xlabel('Número de Vecinos (K)', fontsize=12)
plt.ylabel('Tasa de Error (1 - Precisión)', fontsize=12)
plt.xticks(K_Valores_Grafico)
plt.legend()
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.show()

## 2. Mapa de Calor de Supervivencia por Clase y Género (Datos Reales)
heatmap_data = Registro_Pasajeros.pivot_table(index='Genero', columns='Pclass', values='Sobrevivio', aggfunc='mean')
print("\n--- Datos: Mapa de Calor de Supervivencia (Tasa Media) ---")
print(heatmap_data) # Salida a terminal
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, 
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            linewidths=0.5, 
            linecolor='black',
            cbar_kws={'label': 'Tasa Media de Supervivencia (0.0 a 1.0)'})
plt.title('2. Mapa de Calor de la Tasa de Supervivencia por Género y Clase', fontsize=14)
plt.ylabel('Género', fontsize=12)
plt.xlabel('Clase de Pasajero (Pclass)', fontsize=12)
plt.show()

## 3. Mapa de Calor de la Matriz de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(Matriz, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=['Predicción: No Sobrevive (0)', 'Predicción: Sobrevive (1)'], 
            yticklabels=['Actual: No Sobrevive (0)', 'Actual: Sobrevive (1)'],
            linewidths=0.5,
            linecolor='black')
plt.ylabel('Etiqueta Real')
plt.xlabel(f'Etiqueta Predicha (K={Valor_K_Base})')
plt.title(f'3. Matriz de Confusión (Mapa de Calor, K={Valor_K_Base})', fontsize=14)
plt.show()

## 4. Gráfico de Barras: Conteo de Supervivencia por Género (Datos Totales)
plt.figure(figsize=(8, 6))
sns.countplot(x='Genero', hue='Sobrevivio', data=Registro_Pasajeros, palette='viridis')
plt.title('4. Conteo de Supervivencia por Género (Datos Totales)')
plt.xlabel('Género')
plt.ylabel('Número de Pasajeros')
plt.legend(title='Sobrevivió', labels=['No (0)', 'Sí (1)'])
plt.grid(axis='y', alpha=0.5)
plt.show()

## 5. Gráfico de Barras: Correlación entre Pclass y Supervivencia (Datos Totales)
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Sobrevivio', data=Registro_Pasajeros, palette='Set1')
plt.title('5. Tasa de Supervivencia Media por Clase de Pasajero (Pclass)')
plt.xlabel('Clase de Pasajero (1=Primera, 2=Segunda, 3=Tercera)')
plt.ylabel('Tasa de Supervivencia (%)')
plt.grid(axis='y', alpha=0.5)
plt.show()

# --- GRÁFICAS DE COMPARACIÓN REAL vs. PREDICTO (Usando K Óptimo) --- 

# Creación de DataFrame de comparación
df_comparacion = pd.DataFrame({'Tipo': ['Real'] * len(Y_Prueba) + ['Predicho'] * len(Y_Pred_Base), 
                               'Sobrevivio': np.concatenate([Y_Prueba, Y_Pred_Base])})

## 6. Conteo de Sobrevivientes Reales vs. Predichos (Countplot)
plt.figure(figsize=(8, 6))
sns.countplot(x='Sobrevivio', hue='Tipo', data=df_comparacion, palette=['#1f77b4', '#ff7f0e'])
plt.title(f'6. Comparación de Conteo de Sobrevivientes (Real vs. Predicho) en Test Set (K={Valor_K_Base})')
plt.xlabel('Sobrevivió (0 = No, 1 = Sí)')
plt.ylabel('Número de Pasajeros')
plt.legend(title='Tipo de Dato')
plt.grid(axis='y', alpha=0.5)
plt.show()