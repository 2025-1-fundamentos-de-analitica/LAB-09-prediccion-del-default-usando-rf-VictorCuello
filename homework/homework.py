# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501
import pandas as pd
import numpy as np
import zipfile
import joblib
import json
import os
import pickle
import gzip

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# --- CONFIGURACIÓN GLOBAL DE RUTAS ---
BASE_PATH = "files"
INPUT_PATH = os.path.join(BASE_PATH, "input")
GRADING_PATH = os.path.join(BASE_PATH, "grading") # Para los .pkl intermedios
OUTPUT_PATH = os.path.join(BASE_PATH, "output")
MODELS_PATH = os.path.join(BASE_PATH, "models")

# Crear todos los directorios necesarios para evitar errores
os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(GRADING_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# =============================================================================
# PASO A: PREPARACIÓN DE DATOS Y GENERACIÓN DE ARCHIVOS .pkl
# =============================================================================
print("--- INICIANDO PASO A: Preparación de Datos ---")

# Nombres de los archivos ZIP originales
TRAIN_ZIP_FILE = 'train_data.csv.zip'
TEST_ZIP_FILE = 'test_data.csv.zip'

def load_data_from_zip(zip_path):
    """Carga el primer archivo CSV encontrado dentro de un ZIP en un DataFrame."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
            with z.open(csv_filename) as f:
                return pd.read_csv(f)
    except Exception as e:
        print(f"Error cargando datos desde {zip_path}: {e}")
        return None

def clean_data(df):
    """Aplica las transformaciones de limpieza de datos a un DataFrame."""
    if df is None: return None
    df = df.copy()
    if 'default payment next month' in df.columns:
        df.rename(columns={'default payment next month': 'default'}, inplace=True)
    if 'ID' in df.columns:
        df.drop('ID', axis=1, inplace=True)
    df.dropna(inplace=True)
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)
    pay_cols = [col for col in df.columns if col.startswith('PAY_')]
    for col in pay_cols:
        df[col] = df[col].replace({-2: 0, -1: 0})
    return df

# Cargar los datos crudos
print("Cargando datos desde archivos ZIP...")
df_train_raw = load_data_from_zip(os.path.join(INPUT_PATH, TRAIN_ZIP_FILE))
df_test_raw = load_data_from_zip(os.path.join(INPUT_PATH, TEST_ZIP_FILE))

# Limpiar y dividir los datos
print("Limpiando y dividiendo los datasets...")
train_df_cleaned = clean_data(df_train_raw)
test_df_cleaned = clean_data(df_test_raw)

X_train = train_df_cleaned.drop('default', axis=1)
y_train = train_df_cleaned['default']
X_test = test_df_cleaned.drop('default', axis=1)
y_test = test_df_cleaned['default']

# Guardar los archivos .pkl en la carpeta 'grading' para pasar el test
print(f"Guardando archivos .pkl en la carpeta: {GRADING_PATH}")
X_train.to_pickle(os.path.join(GRADING_PATH, "x_train.pkl"))
y_train.to_pickle(os.path.join(GRADING_PATH, "y_train.pkl"))
X_test.to_pickle(os.path.join(GRADING_PATH, "x_test.pkl"))
y_test.to_pickle(os.path.join(GRADING_PATH, "y_test.pkl"))
print("Archivos .pkl generados exitosamente.")

# =============================================================================
# PASO B: ENTRENAMIENTO DEL MODELO
# =============================================================================
print("\n--- INICIANDO PASO B: Entrenamiento del Modelo ---")

# Los datos (X_train, y_train, etc.) ya están en memoria, los usamos directamente.

print("Creando el pipeline de preprocesamiento y modelado...")
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE'] + [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

print("Iniciando la optimización de hiperparámetros con GridSearchCV...")
param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [10, 20],
    'classifier__min_samples_leaf': [5, 10]
}
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring='balanced_accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\nOptimización completada.")
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor puntuación (balanced_accuracy) en validación cruzada: {grid_search.best_score_:.4f}")

# =============================================================================
# PASO C Y D: GUARDADO DEL MODELO Y EVALUACIÓN
# =============================================================================
print("\n--- INICIANDO PASO C y D: Guardado y Evaluación ---")

# Guardar el modelo final
model_path = os.path.join(MODELS_PATH, "model.pkl.gz")
print(f"Guardando el mejor modelo en: {model_path}")
joblib.dump(best_model, model_path)

# Evaluar el modelo
print("Calculando métricas y matrices de confusión...")
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

def get_metrics_and_cm(y_true, y_pred, dataset_name):
    """Calcula y formatea las métricas y la matriz de confusión."""
    metrics = {
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)}
    }
    return metrics, cm_dict

train_metrics, train_cm = get_metrics_and_cm(y_train, y_train_pred, 'train')
test_metrics, test_cm = get_metrics_and_cm(y_test, y_test_pred, 'test')

# Guardar los resultados en el archivo JSON
all_results = [train_metrics, test_metrics, train_cm, test_cm]
metrics_path = os.path.join(OUTPUT_PATH, "metrics.json")
print(f"Guardando los resultados en: {metrics_path}")
with open(metrics_path, 'w') as f:
    json.dump(all_results, f, indent=4)

print("\n--- PROCESO COMPLETO ---")
print("El script ha finalizado exitosamente.")
print("Todos los artefactos (.pkl, .gz, .json) han sido generados.")