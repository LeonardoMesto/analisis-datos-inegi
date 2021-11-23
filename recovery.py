######################$#####
# Imports # #
#############
##########
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import sweetviz as sv
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_echarts import st_echarts
from sklearn.linear_model import LinearRegression
import plotly.express as px
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#####################
# Initial Variables #
#####################
@st.cache(allow_output_mutation=True)
def get_data(nrows):
    URL = "./db_proyect_clear.csv"
    return pd.read_csv(URL, nrows=nrows)
def get_features():
    URL = "./features.csv"
    return pd.read_csv(URL)
def map(data, lat, lon, zoom):
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position=["longitude", "latitude"],
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
        ]
    ))
def lineal_regression(data):
    return sns.jointplot(x="PDESOCUP_F", y="PCATOLICA", data=data,
                  kind="reg", truncate=False,
                  #xlim=(0, 180000), ylim=(0, 550),
                  color="m", height=7)
df = get_data(100000)
df_index = df.reset_index()
dffeatures = get_features()

################
# Introducción #
################
st.title("Recovery")
st.markdown("""
Se analiza un dataset proporcionado por INEGI, con el objetivo de identificar algunas causas que facilitan o dificultan la educación
en el estado de Jalisco.\n
Las instancias con las que cuenta el dataset, son las características de la población en los distintos municipios del estado de Jalisco.
Se cuenta con 10348 instancias iniciales. \n
Las instancias comparten la relación de ser registros de las poblaciones que forman parte del censo de población y vivienda 
del INEGI en Jalisco.
""")

#############
# Dataframe #
#############
st.dataframe(df.head())

################################
# Descripción de los atributos #
################################
st.subheader("Descripción de los atributos")
st.markdown("""
En este apartado, podremos observar el dataset con sus distintos atributos (columnas),
de los cuales, todos son datos estructurados.\n
Si desea conocer más sobre cada una de las columnas proporcionadas por el dataset, puede
seleccionar el atributo en las opciones posteriores.
""")

#########################
# Catálogo de variables #
#########################
searched_Name = dffeatures.groupby('Mnemonico')['Mnemonico'].count()\
    .sort_values(ascending=False).index
select_name = []
select_name.append(st.selectbox('', searched_Name))
name_df = dffeatures[dffeatures['Mnemonico'].isin(select_name)]
st.markdown(f"***Indicador:*** {name_df['Indicador'].values[0]}.")
st.markdown(f"***Descripción:*** {name_df['Descripcion'].values[0]}")
st.markdown(f"***Tipo de dato:*** {name_df['Tipo'].values[0]}.")

#######################
# Limpieza de valores #
#######################
st.subheader("Limpieza de valores")
st.markdown("""
El manejo de datos nulos que se siguió fue eliminar las filas en donde se encontraban datos nulos, porque se identificó un
patrón donde los datos nulos siempre se mostraban varias veces a lo largo de los registros; esto impedía obtener los datos
necesarios para el análisis y, por lo tanto, se volvía obsoleto el registro. 
""")

df = df.replace({"*": np.nan})
df = df.dropna(axis = 0).reset_index(drop=True)
st.dataframe(df.head(20))

#################
# Mapa de calor #
#################
st.markdown("""
Posterior a la limpieza de los datos, se decidió realizar un mapa de calor, con la finalidad de conocer las variables que
convergen con mayor impacto, para poder tener una forma visual, a fin de analizar los cruces más importantes.\n
El código para poder desarrollar el mapa de calor se realizó utilizando [Sweetviz](https://pypi.org/project/sweetviz/) de la siguiente
manera.
""")
st.code("""
feature_config = sv.FeatureConfig(skip=["MUN", "LOC", "NOM_LOC", "LATITUD", "LONGITUD", "ENTIDAD", "NOM_ENT", "ALTITUD", "POBTOT", "VIVTOT", "TAMLOC", "TVIVHAB", "P3HLINHE_F", "P3HLINHE_M", "PDER_ISTEE", "PDER_IMSSB", "PAFIL_PDOM", "VPH_S_ELEC", "VPH_AGUAFV", "VPH_LETR", "VPH_NODREN"])
report = sv.analyze(df, feat_cfg=feature_config, pairwise_analysis="on")

report.show_html()
""")
st.markdown("""
En la primera línea del código podemos identificar que se decidió descartar algunas variables, esto es porque dichas variables no generaban ningún cruce en el mapa de calor (como Municipio), o simplemente no tenían relevancia al realizar los cruces (como Latitud y Longitud). Posteriormente se realizaron ciertas configuraciones al reporte y el resultado rescatado fue el siguiente.
""")

st.image('mapa_calor.jpg')

############
# Gráficas #
############
#----------------Descriptive----------------

st.header("Análisis Descriptivo")
searched_Location = set(df['NOM_MUN'])
st.markdown("### **Selecciona un Municipio**")
st.markdown("""
En esta sección se muestra por porcentajes a la población total de cada municipio basado en su edad, clasificado en tres secciones distintas, infantes con edades de 0 a 14 años, adultos con edades de 15 a 64 años y adultos mayores con edades por encima de los 65 años.
""")

select_mun = []
select_mun.append(st.selectbox('', searched_Location))
mun_df = df[df['NOM_MUN'].isin(select_mun)]

total = 0
infantes = 0
adultos = 0
survivors = 0

for i in range(len(mun_df['POBTOT'])):
    total = total + int(mun_df['POBTOT'].values[i])

for i in range(len(mun_df['POB0_14'])):
    infantes = infantes + int(mun_df['POB0_14'].values[i])

for i in range(len(mun_df['POB15_64'])):
    adultos = adultos + int(mun_df['POB15_64'].values[i])

for i in range(len(mun_df['POB65_MAS'])):
    survivors = survivors + int(mun_df['POB65_MAS'].values[i])

st.markdown(f"**Total de Población: ** {total}")
st.markdown(f"**Infantes (0-14 años):  ** {infantes} ")
st.markdown(f"**Adultos (15-64 años):  ** {adultos} ")
st.markdown(f"**Mayores (65++ años):  ** {survivors} ")
#Pie Chart - Survival of passangers
st.write(" ")
st.subheader(f"Edades en el municipio de {mun_df['NOM_MUN'].values[0]}")
labels = 'Infantes', 'Adultos', 'Mayores'
sizes = [infantes, adultos, survivors]
explode = (0.025, 0.025, 0.025)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig1)


#-------------------------------------------
st.header("Cantidad de personas Analfabetas por Municipio")
st.write("En estas gráficas se utilizaron varias columnas del DataFrame, como Nombre de los Municipios, Localidades de los municipios y Población de 8 a 14 años que no sabe leer y escribir, ¿Qué es lo que podemos hacer con esta gráfica? Nos sirve para poder comparar entre Municipios para así saber el número de personas analfabetas dentro de cada localidad del Municipio.")
all_mun = df_index.NOM_MUN.unique().tolist()
defa = ["Amatitan"]
mun= (st.multiselect("Municipios", options=all_mun, default=defa))
plot_df = df_index[df_index.NOM_MUN.isin(mun)]
plot_df["P8A14AN"] = plot_df.P8A14AN

chart = (
    alt.Chart(
        plot_df
    )
    .mark_bar()
    .encode(
        x=alt.X("index", title="Poblacion de 8 a 14 años que no sabe leer y escribir"),
        y=alt.Y(
            "NOM_LOC",
            sort=alt.EncodingSortField(field="NOM_LOC",order="ascending"),
            title="Localidades",
        ),
        color=alt.Color(
            "NOM_MUN",
            legend=alt.Legend(title="Municipios"),
            scale=alt.Scale(scheme="category10"),
        ),
        tooltip=["NOM_LOC", "P8A14AN"],
    )
)
st.altair_chart(chart, use_container_width=True)

#----------------Diagnóstico----------------

st.header("Análisis de Diagnóstico")
st.subheader("Distribución de población con respecto a la religión católica")
st.markdown("""
Un factor interesante que se rescató del mapa de calor fue la relación que tiene gran parte de la población que no asiste a la escuela y no sabe leer y escribir, con respecto a su creencia católica, donde se puede tomar a consideración como una correlación, ya que se en varias zonas (comúnmente rurales), se tienen más o menos oportunidades que algunas zonas, y por ello las prioridades de las personas llegan a cambiar, en la que pueden tomar como prioridad la religión antes que el estudio, lo cual podremos apreciar en los mapas siguientes.\n
Primeramente, se muestran varios botones donde podrás seleccionar una de las categorías, y con respecto a la opción seleccionada, los mapas reaccionarán y mostrarán los puntos en donde se presentan registros. Por ejemplo, al seleccionar la opción de "Población femenina de 6 y 11 años que no asiste a la escuela", se mostrarán en los mapas las diferentes zonas en las que esta población es católica.
""")

values = {
    0 : "P6A11_NOAF",
    1 : "P6A11_NOAM",
    2 : "P12A14NOAF",
    3 : "P18A24A",
    4 : "P8A14AN_F",
    5 : "P8A14AN_M"
}

dfcategories = [
    "Población femenina de 6 a 11 años que no asiste a la escuela.",
    "Población masculina de 6 a 11 años que no asiste a la escuela.",
    "Población femenina de 12 a 14 años que no asiste a la escuela.",
    "Población de 18 a 24 años que asiste a la escuela.",
    "Población femenina de 8 a 14 años que no sabe leer y escribir." ,
    "Población masculina de 8 a 14 años que no sabe leer y escribir."
]

categories = st.radio("Categorías", dfcategories)

st.markdown("""
El siguiente mapa demostrará por medio de puntos, todas las zonas en las que concuerda la Categoría (como Población femenina de 6 a 11 años que no asiste a la escuela) y que a su vez son personas católicas.
""")

st.map(df.query(f"{values[dfcategories.index(categories)]}>=0")[['latitude', 'longitude']])

st.markdown("""
El siguiente mapa demostrará por medio de puntos, todas las zonas en las que concuerda la Categoría (como Población femenina de 6 a 11 años que no asiste a la escuela) y que a su vez son personas católicas, donde los puntos pueden crecer en tamaño si es el caso de que más de 1 registro concuerda con la zona (por ejemplo, 10 personas cumplen con la condición y están en la misma zona de Tlajomulco).
""")

df = df.query(f"{values[dfcategories.index(categories)]}>0")

midpoint = (np.average(df["latitude"]), np.average(df["longitude"]))

map(df, midpoint[0], midpoint[1], 11)

st.header("Análisis Predictivo")
st.subheader("Regresión Lineal - Poblacion Desocupada y Católica")
st.markdown("""
En este punto, se observó que existe una correlación fuerte en específico entre la *Población Femenina Desocupada* y la *Población Católica*, por lo que se decidió poder analizar de mejor manera la relación entre estos dos campos, donde se puede observar en la gráfica siguiente que la relación entre ambos tiende a ser una relación lineal.
""")

y=np.array(df["PCATOLICA"])
x=np.array(df["PDESOCUP_F"])

fig = px.scatter(
        x=x,
        y=y
    )
fig.update_layout(
    yaxis_title="Población Católica",
    xaxis_title="Población Desocupada Femenina",
    title="Población Desocupada y Católica"
)

st.write(fig)

st.markdown("""
Después de observar que los puntos marcan una tendencia lineal, se realizó una *Regresión Lineal* para poder conocer el modelo al que tiende esta relación.
""")

st.image('lineal_regression.jpg')
#st.pyplot(lineal_regression(df))

st.markdown("""
Al obtener el modelo de la relación, se podrán predecir el número de población católica dependiendo del municipio de Jalisco al que seleccione, utilizando una función interactiva donde podrá seleccionar el estado y le mostraremos el número de *Población Católica* con respecto al número de *Población Desocupada Femenina*.
""")

mun = df_index.NOM_MUN.unique().tolist()
select_name = []
select_name.append(st.selectbox('', mun))
model = LinearRegression()
model.fit(np.array(y).reshape(-1, 1),np.array(x))

name_mun = df_index[df_index['NOM_MUN'].isin(select_name)]
weight_t = model.predict(np.array([np.sum(name_mun['PCATOLICA'])]).reshape(-1, 1))

st.markdown(f"En el municipio de ***{select_name[0]}*** existen un total de ***{np.sum(name_mun['PCATOLICA'])}*** Personas Católicas, habrán una Población Femenina Desocupada de ***{np.sum(name_mun['PDESOCUP_F'])}***.")

st.markdown("### Predicción personalizada")
st.markdown("""
En el caso de que deseé hacer una predicción con otro valor de la Población Católica para conocer la Población Desocupada Femenina, puede ingresar el número a predecir y en la parte inferior se le mostrará el resultado obtenido.
""")
message = st.number_input(label="Elige el número de Personas Católicas para predecir el número de Población Femenina Desocupada", format="%i", value=200000)
st.markdown(f"De ***{message}*** Personas Católicas, se estima que habrán una Población Femenina Desocupada de ***{model.predict(np.array(message).reshape(-1, 1))[0].astype(int)}*** personas.")

#----------------Prescriptivo----------------

st.header("Análisis de Prescriptivo")
st.markdown("""
De acuerdo con los análisis anteriores, se pudo visualizar que hay una relación entre la población que no trabajan ni estudian, con la población que es católica, por lo que se hace la hipótesis que posiblemente una de las razones por las que algunas personas no trabajen ni estudien, se deba a la influencia de la religión.\n
Las acciones sugeridas a llevar a cabo son:\n
- Que la religión encuentre lo valioso en la educación y los sermones de los sacerdotes se enfoquen en la importancia de la educación y trabajo en la vida.
- Que a los niños se les de clases de catecismo, y a la vez, se realicen cursos educativos.
- Para los adultos, poder tener las catequesis, agregando talleres para enseñar herramientas útiles para la vida laboral.
""")