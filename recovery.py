## Imports
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import seaborn as sns
import sweetviz as sv
import numpy as np
from streamlit_echarts import st_echarts
import plotly.express as px

## Header
st.title("Recovery")
st.markdown("""
Se analiza un dataset proporcionado por INEGI, con el objetivo de identificar algunas causas que facilitan o dificultan la educación
en el estado de Jalisco.\n
Las instancias con las que cuenta el dataset, son las características de la población en los distintos municipios del estado de Jalisco.
Se cuenta con 10348 instancias iniciales. \n
Las instancias comparten la relación de ser registros de las poblaciones que forman parte del censo de población y vivienda 
del INEGI en Jalisco.
""")

# Initial Variables
@st.cache(allow_output_mutation=True)
def get_data():
    URL = "./db_proyecto.csv"
    return pd.read_csv(URL)
def get_features():
    URL = "./features.csv"
    return pd.read_csv(URL)

df = get_data()
dffeatures = get_features()
#df['Age'] = df['Age'].astype(int)
#df['Survived'] = df['Survived'] == 1
#N = 8

##############
#  Dataframe #
##############
st.dataframe(df.head())

##########################
# Step 1 - Column Filter #
##########################
## Description
st.subheader("Descripción de los atributos")
st.markdown("""
En este apartado, podremos observar el dataset con sus distintos atributos (columnas),
de los cuales, todos son datos estructurados.\n
Si desea conocer más sobre cada una de las columnas proporcionadas por el dataset, puede
seleccionar el atributo en las opciones posteriores.
""")

#------------------------Module 2--------------------------
searched_Name = dffeatures.groupby('Mnemonico')['Mnemonico'].count()\
    .sort_values(ascending=False).index
select_name = []
select_name.append(st.selectbox('', searched_Name))
name_df = dffeatures[dffeatures['Mnemonico'].isin(select_name)]
st.markdown(f"***Indicador:*** {name_df['Indicador'].values[0]}.")
st.markdown(f"***Descripción:*** {name_df['Descripcion'].values[0]}")
st.markdown(f"***Tipo de dato:*** {name_df['Tipo'].values[0]}.")

#####################
# Limpieza de datos #
#####################
st.subheader("Limpieza de valores")
st.markdown("""
El manejo de datos nulos que se siguió fue eliminar las filas en donde se encontraban datos nulos, porque se identificó un
patrón donde los datos nulos siempre se mostraban varias veces a lo largo de los registros; esto impedía obtener los datos
necesarios para el análisis y, por lo tanto, se volvía obsoleto el registro. 
""")

df = df.replace({"*": np.nan})
df = df.dropna(axis = 0).reset_index(drop=True)
st.dataframe(df.head(20))
feature_config = sv.FeatureConfig(skip=["MUN", "LOC", "NOM_LOC", "LATITUD", "LONGITUD", "ENTIDAD", "NOM_ENT", "ALTITUD", "POBTOT", "VIVTOT", "TAMLOC", "TVIVHAB"])
report = sv.analyze(df, feat_cfg=feature_config, pairwise_analysis="on")
file_name = "Recovery_report.html"
o_b = False
report.show_html(file_name, open_browser = o_b)