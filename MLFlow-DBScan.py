import mlflow
import pandas as pd
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Cargar tus datos (en este caso, se asume que ya tienes "X_pca" cargado)
#Cargamos los datos
df = pd.read_csv('Proyecto_DSA/data/online_shoppers_intention.csv', sep=',')
df.Weekend = df.Weekend.replace({True: 1, False: 0})
df.Revenue = df.Revenue.replace({True: 1, False: 0})

df.Month = df.Month.replace({'Feb': 2, 'Mar': 3, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov':11, 'Dec': 12})
# Tenemos variables categoricas, por lo cual debemos crear la dummies correspondientes
cat_cols = ['VisitorType']
dummies = pd.get_dummies(df[cat_cols], prefix=cat_cols)

# Concatenamos los datos originales con las variables ficticias
df_con_dummies = pd.concat([df, dummies], axis=1)

# Eliminamos las variables originales ya que ahora tenemos las variables ficticias
df_con_dummies = df_con_dummies.drop(cat_cols, axis=1)

# Variables numericas
val_num = df_con_dummies.select_dtypes(include=["number"])

# Estandarizamos las variables
X = val_num.values

# Estandarizamos los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce la dimensionalidad a 2D utilizando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Configurar el experimento en MLflow con el nombre "DBScan"
experiment_name = "DBScan"
mlflow.set_experiment(experiment_name)

# Definir los parametros (puedes ajustar esto según tus necesidades)
param_min_samples = 3
param_eps = 0.5

# Crear y entrenar el modelo DBScan
model = DBSCAN(eps=param_eps, min_samples=param_min_samples)
model.fit(X_pca)

# Predecir las etiquetas de los clústeres
labels = model.labels_

# Calcular el coeficiente de silueta
silhouette_avg = silhouette_score(X_pca, labels)

# Lograr los resultados en MLflow
with mlflow.start_run():
    # Lograr el nombre del experimento y su ID
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.log_param("Experiment Name", experiment_name)
    mlflow.log_param("Experiment ID", experiment_id)

    # Lograr los resultados del coeficiente de silueta
    mlflow.log_param("Min Samples", param_min_samples)
    mlflow.log_param("Eps", param_eps)
    mlflow.log_metric("Silhouette Score", silhouette_avg)

    # Lograr el modelo (opcional, pero puedes querer hacer esto para reproducibilidad)
    mlflow.sklearn.log_model(model, "DBScan_Model")

# Finalmente, imprime el enlace a la interfaz web de MLflow
print(f"Experiment URL: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}")

mlruns_location = mlflow.get_tracking_uri()
print(f"Ubicación de mlruns: {mlruns_location}")
