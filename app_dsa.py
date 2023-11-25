import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
#import mlflow
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids

# Cargar tus datos y procesarlos (suponiendo que ya tienes 'X_pca' cargado)
# ... (código para cargar y procesar datos)

# Código para el modelo KMeans y KMedoids
# ...

# Configuración de la app Dash
app = dash.Dash(__name__)


###
# Código kmeans
###
import pandas as pd
import os
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

# Configurar el experimento en MLflow con el nombre "Kmeans"
experiment_name = "Kmeans"

# Definir el número de clústeres (puedes ajustar esto según tus necesidades)
num_clusters = 3


# Crear y entrenar el modelo KMeans
model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
model.fit(X_pca)

# Asignar los clusters a tus datos
clusters = model.predict(X_pca)

# Incorporar la información de los clusters al DataFrame original
df_con_dummies['Cluster'] = clusters

# Crear el DataFrame que se usará para la visualización
data_for_plot = pd.DataFrame({
    'X': X_pca[:, 0],  # Coordenada X del PCA
    'Y': X_pca[:, 1],  # Coordenada Y del PCA
    'Cluster': clusters  # Información de los clusters
})

# Crear el gráfico de dispersión con Plotly Express
scatter_fig = px.scatter(
    data_for_plot,
    x='X',
    y='Y',
    color='Cluster',
    title='Gráfico de dispersión con ' + str(num_clusters) + ' Clusters',
    width=800,
    height=600
)


df = pd.read_csv('Predicciones.csv')
# Eliminar duplicados en la columna 'Month'
unique_months = df['Month'].drop_duplicates().sort_values()
unique_sd = df['SpecialDay'].drop_duplicates().sort_values()
unique_region = df['Region'].drop_duplicates().sort_values()
unique_weekend = df['Weekend'].drop_duplicates().sort_values()
unique_revenue = df['Revenue'].drop_duplicates().sort_values()
unique_tt = df['TrafficType'].drop_duplicates().sort_values()
unique_browser = df['Browser'].drop_duplicates().sort_values()
unique_info = df['Informational'].drop_duplicates().sort_values()
unique_admin = df['Administrative'].drop_duplicates().sort_values()



# Generar las opciones del Dropdown a partir de los valores únicos de 'Month'
dropdown_options = [{'label': month, 'value': month} for month in unique_months]
dropdown_options1 = [{'label': month, 'value': month} for month in unique_sd]
dropdown_options2 = [{'label': month, 'value': month} for month in unique_region]
dropdown_options3 = [{'label': month, 'value': month} for month in unique_weekend]
dropdown_options4 = [{'label': month, 'value': month} for month in unique_revenue]
dropdown_options5 = [{'label': month, 'value': month} for month in unique_tt]
dropdown_options6 = [{'label': month, 'value': month} for month in unique_browser]
dropdown_options7 = [{'label': month, 'value': month} for month in unique_info]
dropdown_options8 = [{'label': month, 'value': month} for month in unique_admin]






###
# Mostrar el gráfico
#scatter_fig.show()
###

# Define el layout del dashboard
app.layout = html.Div(
    id="app-container",
    children=[
        html.Div(
            # Sección superior con título centrado
            html.H1(children="Segmentación de Clientes Compras Online", style={"text-align": "left", "margin-top": "20px", "margin-left": "30px", "margin-bottom": "0px"})
        ),

        # Columna izquierda
        html.Div(
            id="left-column",
            className="four columns",
            children=[
                html.Div(
                    id="description-card",
                    children=[
                        html.H6("Esta herramienta permite identificar a qué segmento de cliente puede pertenecer un usuario, de acuerdo con las características que seleccione.", style={"text-align": "justify"}),
                        html.Br(),

						
    html.P("Month"),
	dcc.Dropdown(
    id='dropdown-month',
    options=dropdown_options,
    value=unique_months.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),

html.Br(),
	html.P("SpecialDay"),
dcc.Dropdown(
    id='dropdown-special-day',
    options=dropdown_options1,
    value=unique_sd.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),


html.Br(),
	html.P("Region"),
dcc.Dropdown(
    id='dropdown-region',
    options=dropdown_options2,
    value=unique_region.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),


html.Br(),
	html.P("Weekend"),
dcc.Dropdown(
    id='dropdown-weekend',
    options=dropdown_options3,
    value=unique_weekend.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),


html.Br(),
	html.P("Revenue"),
dcc.Dropdown(
    id='dropdown-revenue',
    options=dropdown_options4,
    value=unique_revenue.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),


html.Br(),
	html.P("TrafficType"),
dcc.Dropdown(
    id='dropdown-tt',
    options=dropdown_options5,
    value=unique_tt.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),


html.Br(),
	html.P("Browser"),
dcc.Dropdown(
    id='dropdown-browser',
    options=dropdown_options6,
    value=unique_browser.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),


html.Br(),
html.P("Informational"),
dcc.Dropdown(
    id='dropdown-info',
    options=dropdown_options7,
    value=unique_info.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),


html.Br(),
	html.P("Administrative"),
dcc.Dropdown(
    id='dropdown-admin',
    options=dropdown_options8,
    value=unique_admin.iloc[0]  # Valor predeterminado para la lista desplegable
 # Valor predeterminado para la lista desplegable
    ),

    
  html.Div(id='display-selected'),
	




                    ],
                ),
            ],
        ),

        # Columna derecha
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # Tarjetas con los valores de los clusters y porcentaje de coincidencia
                html.Div(
                    id="cluster-values",
                    style={"display": "flex", "justify-content": "space-around", "flex-wrap": "nowrap", "overflow": "auto"},
                    children=[
                        html.Div(
                            style={"flex": "0 0 auto", "width": "40%"},  # Mantiene el ancho al 50%
                            children=[
                                html.H2("2", style={"background-color": "#eef2f2", "border-radius": "10px", "text-align": "center", "font-size": "50px", "margin": "20px 60px 0px 60px", "padding": "20px"}),
                                html.H6("Número Clúster", style={"text-align": "center", "margin": "0px 60px 0px 60px", "padding": "0px", "page-break-after": "always", "display": "inline-block"}),
                            ],
                        ),
                        html.Div(
                            style={"flex": "0 0 auto", "width": "40%"},  # Mantiene el ancho al 50%
                            children=[
                                html.H2(html.Div(id='display-selected') , style={"background-color": "#eef2f2", "border-radius": "10px", "text-align": "center", "font-size": "50px", "margin": "20px 60px 0px 60px", "padding": "20px"}),
                                html.H6("Porcentaje Coincidencia", style={"text-align": "center", "margin": "0px 60px 0px 60px", "padding": "0px", "page-break-after": "always", "display": "inline-block"}),
                            ],
                        ),
                    ],
                ),

                html.Div(
                    style={"flex": "0 0 auto", "width": "100%"},  # Mantiene el ancho al 100%
                    children=[
                        html.H4("Descripción del Clúster", style={"text-align": "center", "margin": "20px 0px 0px 0px", "padding": "0px", "page-break-after": "always", "display": "inline-block"}),
                        html.H6("De acuerdo con los resultados obtenidos, se ha asignado al usuario al clúster dada la probabilidad indicada en la parte superior.", style={"text-align": "justify"}),
                        html.Br(),
                    ],
                ),

                # Gráfico de dispersión
                dcc.Graph(figure=scatter_fig)
            ],
        ),
    ],
)


# Callback para actualizar el gráfico de dispersión según el número de clusters seleccionado
#@app.callback(
#    Output("scatter_fig", "figure"),
#    [Input("slider-cluster-1", "value")]
#)
#def update_scatter_plot(num_clusters):
#    # Código para el modelo KMeans con el número de clusters seleccionado
#    # ... (código para cargar y procesar datos)
#   # ... (modelo KMeans con el número de clusters)
#    # ... (gráfico de dispersión)

#    # Utilizando scatter_fig como ejemplo, reemplaza este código con la lógica para generar el gráfico de dispersión
#    scatter_fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])  # Reemplaza esto con tu gráfico de dispersión
#
 #   return scatter_fig
#@app.callback(
#    Output('display-selected', 'children'),
#    [Input('dropdown', 'value')]
#)
#def update_output(selected_value):
#    return f'Selección actual: {selected_value}'


# Callback para actualizar el resultado según la selección de dropdowns
@app.callback(
    Output('display-selected', 'children'),
    [Input('dropdown-month', 'value'),
     Input('dropdown-special-day', 'value'),
     Input('dropdown-region', 'value'),
     Input('dropdown-weekend', 'value'),
     Input('dropdown-revenue', 'value'),
     Input('dropdown-tt', 'value'),
     Input('dropdown-browser', 'value'),
     Input('dropdown-info', 'value'),
     Input('dropdown-admin', 'value')]
)
def update_output(selected_month, selected_sd, selected_region, selected_weekend, selected_revenue, selected_tt, selected_browser, selected_info, selected_admin):
    # Filtrar el archivo 'Predicciones.csv' según las selecciones
    # (reemplaza la lógica con la estructura y los datos de tu archivo 'Predicciones.csv')
    filtered_data = df[(df['Month'] == selected_month) & (df['SpecialDay'] == selected_sd) & (df['Region'] == selected_region) & (df['Weekend'] == selected_weekend) & (df['Revenue'] == selected_revenue) & (df['TrafficType'] == selected_tt) & (df['Browser'] == selected_browser) & (df['Informational'] == selected_info) & (df['Administrative'] == selected_admin)]
    
    # Procesa la información filtrada y devuelve el resultado deseado
    return (str(round((filtered_data["Confianza"][0]), 3) * 100), '%')  # Actualiza con los valores filtrados

# Ejecutar el servidor
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8051, debug=True)


