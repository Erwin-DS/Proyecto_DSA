import mlflow
import pandas as pd
import mlflow.sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Cargar tus datos (en este caso, se asume que ya tienes "X_pca" cargado)
#Cargamos los datos
df = pd.read_csv('data/online_shoppers_intention.csv', sep=',')
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

def train_evaluate_kmeans(X, n_clusters, run_name="MLflow KMeans"):
<<<<<<< HEAD
    # Iniciamos una corrida de MLflow
   with mlflow.start_run(run_name=run_name) as run:
    
    # MLflow asigna un ID al experimento y a la corrida
    experiment_id = run.info.experiment_id
    run_id = run.info.run_uuid

    # Log de parámetros en MLflow
    mlflow.log_param("n_clusters", n_clusters)
    # Crear el modelo K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
=======
    with mlflow.start_run(run_name=run_name):
        # Log de parámetros en MLflow
        mlflow.log_param("n_clusters", n_clusters)

        # Crear el modelo K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        # Calcular la puntuación de silhouette
        silhouette_avg = silhouette_score(X, kmeans.labels_)
>>>>>>> origin/master

        # Log de métricas en MLflow
        mlflow.log_metric("silhouette_score", silhouette_avg)

<<<<<<< HEAD
    # Log de métricas en MLflow
    mlflow.log_metric("silhouette_score", silhouette_avg)

    # Guardar el modelo en MLflow
    mlflow.sklearn.log_model(kmeans, "model")
    
    return experiment_id, run_id
=======
        # Guardar el modelo en MLflow
        mlflow.sklearn.log_model(kmeans, "model")
        
        return mlflow.active_run().info.experiment_id, mlflow.active_run().info.run_id
>>>>>>> origin/master

# Ejecutar la función con diferentes valores de clusters
for n_clusters in [2, 3, 4, 5]:
    experiment_id, run_id = train_evaluate_kmeans(X_pca, n_clusters)
    print("MLflow Run completed with run_id {} and experiment_id {}".format(run_id, experiment_id))
