import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Conectar al servidor de AWS
mlflow.set_tracking_uri("http://3.235.99.156:5000")

# Definir el experimento para tu SKU
mlflow.set_experiment("Forecasting_SKU_455151")


## 2. Carga de datos

# 1. Cargar datos (Ruta del dvc sincronizado con datos)
df = pd.read_csv('DATA/raw/ventashistoricas_455151.csv')

# --- MOCK DATA PARA PRUEBAS (Borra esto y usa la línea de arriba) ---
fechas = pd.date_range(start='2023-05-01', end='2026-06-01', freq='D')
df = pd.DataFrame({'Date': fechas, 'Quantity': np.random.poisson(lam=15, size=len(fechas))})
# ---------------------------------------------------------------------

# 2. Preparar el índice de fechas
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').set_index('Date')

# 3. Crear variables de calendario
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 4. Crear variables de rezagos (lags) y promedios móviles
# ¿Cuánto vendimos ayer? ¿Y hace una semana?
df['lag_1'] = df['Quantity'].shift(1)
df['lag_7'] = df['Quantity'].shift(7)
df['rolling_mean_7'] = df['Quantity'].shift(1).rolling(window=7).mean()

# Eliminar valores nulos generados por los shifts
df.dropna(inplace=True)

print(df.head())


## 3. División de datos en entrenamiento y prueba

# Definir cuántos días queremos predecir/validar
dias_test = 30

# Separar variables predictoras (X) de la variable objetivo (y)
X = df.drop(columns=['Quantity'])
y = df['Quantity']

# División cronológica
X_train = X.iloc[:-dias_test]
X_test = X.iloc[-dias_test:]
y_train = y.iloc[:-dias_test]
y_test = y.iloc[-dias_test:]

print(f"Tamaño de entrenamiento: {X_train.shape[0]} días")
print(f"Tamaño de prueba: {X_test.shape[0]} días")


## 4. Modelo BAse - Promeido movil

# Baseline: Predecir usando el promedio móvil de los últimos 7 días
baseline_preds = X_test['rolling_mean_7']

# Baseline: Predecir usando el promedio móvil
baseline_preds = X_test['rolling_mean_7']

mae_base = mean_absolute_error(y_test, baseline_preds)
rmse_base = root_mean_squared_error(y_test, baseline_preds)
mape_base = mean_absolute_percentage_error(y_test, baseline_preds)

print("--- MÉTRICAS DEL BASELINE ---")
print(f"MAE: {mae_base:.2f} | RMSE: {rmse_base:.2f} | MAPE: {mape_base:.2f}")




## 5. ENtrenar y registrar en MLflow

def entrenar_y_registrar(nombre_modelo, modelo, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=nombre_modelo):
        # 1. Entrenar
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        
        # 2. Calcular métricas
        mae = mean_absolute_error(y_test, predicciones)
        # CAMBIO AQUÍ: Usa root_mean_squared_error en lugar de mean_squared_error con squared=False
        rmse = root_mean_squared_error(y_test, predicciones)
        mape = mean_absolute_percentage_error(y_test, predicciones)
        
        # 3. Registrar
        mlflow.log_params(modelo.get_params())
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)
        mlflow.sklearn.log_model(modelo, f"modelo_{nombre_modelo.lower()}")
        
        print(f"{nombre_modelo} -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}")
        return modelo, predicciones

# Diccionario con los 3 modelos a evaluar y sus hiperparámetros iniciales
modelos = {
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
}

# Bucle para entrenar y registrar los 3 modelos
print("\n--- RESULTADOS MODELOS AVANZADOS ---")
resultados_preds = {}
for nombre, modelo_instanciado in modelos.items():
    modelo_entrenado, preds = entrenar_y_registrar(
        nombre, 
        modelo_instanciado, 
        X_train, y_train, X_test, y_test
    )
    resultados_preds[nombre] = preds
