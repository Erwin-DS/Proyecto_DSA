import mlflow
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

# Cargar tus datos (en este caso, se asume que ya tienes "X_pca" cargado)
# X_pca = ...

# Función para entrenar y evaluar el modelo K-means
def train_evaluate_kmeans(X, n_clusters):
    # Crear el modelo K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Calcular la puntuación de silhouette
    silhouette_avg = silhouette_score(X, kmeans.labels_)

    # Obtener el ID del experimento actual
    experiment_id = mlflow.get_experiment_by_name(mlflow.active_run().info.experiment_name).experiment_id

    # Log de parámetros y métricas en MLflow
    with mlflow.start_run():
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("silhouette_score", silhouette_avg)

        # Guardar el modelo en MLflow
        mlflow.sklearn.log_model(kmeans, "model")

        # Obtener el ID de la corrida actual
        run_id = mlflow.active_run().info.run_id

    # Finalizar la ejecución actual
    mlflow.end_run()

    return experiment_id, run_id

# Ejecutar la función con diferentes valores de clusters
for n_clusters in [2, 3, 4, 5]:
    experiment_id, run_id = train_evaluate_kmeans(X_pca, n_clusters)
    print("MLflow Run completed with run_id {} and experiment_id {}".format(run_id, experiment_id))

