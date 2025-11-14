# app.py
import streamlit as st
import os
import tensorflow as tf
import joblib
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
import joblib
import ast

st.set_page_config(page_title="🎥 Proyecto de Ciencia de Datos", layout="wide")



# ===========================
# 📥 Carga de Datos
# ===========================
@st.cache_data
def load_data():
    df_movies = pd.read_csv("datalake/bronze/tmdb_5000_movies.csv").head(10)
    df_credits = pd.read_csv("datalake/bronze/tmdb_5000_credits.csv").head(10)
    df = pd.read_csv("datalake/gold/data.csv")
    return df, df_movies, df_credits

df, df_movies, df_credits = load_data()

# === Cargar JSON con resultados ===
with open("model/metrics.json") as f:
    metrics_data = json.load(f)

# Convertir a DataFrame
df_metrics = pd.DataFrame(metrics_data)

st.sidebar.title("📊 Menú")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🎬 Contexto", "📈 Análisis Exploratorio", "👀 Comparativa", "🤖 Evaluador y predictor"],
    label_visibility="collapsed"
)


if page == "🎬 Contexto":
    st.title("🎬 Contexto")
    st.markdown("""
            # Predicción del Éxito Comercial de Películas

## 1. Objetivo del Proyecto

El objetivo de este proyecto es **predecir la probabilidad de éxito comercial de una película** y **estimar su recaudación esperada**, utilizando información histórica y modelos de aprendizaje automático.

El análisis busca responder preguntas clave como:

- ¿Qué variables influyen más en el éxito de una película?
- ¿Qué nivel de exactitud puede alcanzarse al predecir el éxito antes del estreno?
- ¿Qué tan confiable es la estimación de recaudación en función del presupuesto y otros factores?
---
## 2. Objetivo del Análisis

Desarrollar un sistema de predicción dual que:

- **Clasifique** si una película tiene alta probabilidad de ser exitosa (modelo de clasificación).
- **Estime** su recaudación proyectada en dólares (modelo de regresión).

Estos resultados pueden asistir a **productores, estudios y analistas financieros** en la toma de decisiones estratégicas relacionadas con inversión, marketing y distribución.

---
## 3. Metodología

Se utilizó el dataset público de **The Movie Database (TMDB)**, con más de **5,000 registros de películas**.

Cada película fue descrita mediante variables **numéricas** y **categóricas**:

- **Numéricas**: presupuesto, popularidad, calificación promedio y duración.
- **Categóricas**: género principal y director.
            """)
    st.subheader("Datos que utilizamos")
    tab1, tab2, tab3 = st.tabs(["Películas", "Créditos", "Data Limpia"])

    with tab1:
        st.dataframe(df_movies)
    with tab2:
        st.dataframe(df_credits)
    with tab3:
        st.dataframe(df.head(10))

    st.markdown("---")
    st.markdown("""
## 4. Tecnologías:
    - 🐼 Pandas
    - 🤖 Scikit-learn
    - 📊 Plotly
    - 🚀 Streamlit
            """)

# ===========================
# 📊 Análisis Exploratorio
# ===========================

if page == "📈 Análisis Exploratorio":
    st.header("📈 Análisis Exploratorio")
    st.sidebar.markdown("---")
    selected_genre = st.sidebar.selectbox("Filtrar por género", options=["Todos"] + sorted(df['main_genre'].dropna().unique().tolist()))
    if selected_genre != "Todos":
        df = df[df['main_genre'] == selected_genre]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Presupuesto vs Recaudación")
        fig, ax = plt.subplots()
        sns.scatterplot(x='budget', y='revenue', data=df, alpha=0.6, ax=ax)
        plt.title("Presupuesto vs Recaudación")
        st.pyplot(fig)

    with col2:
        st.subheader("Promedio de Calificación por Década")
        df['decade'] = (df['year'] // 10) * 10
        fig, ax = plt.subplots()
        sns.barplot(x='decade', y='vote_average', data=df, ax=ax)
        plt.title("Calificación promedio por década")
        st.pyplot(fig)


    col3,col4 = st.columns(2)
    with col3:
        st.subheader("Conteo de valores por columna")
        col_counts = df.count().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=col_counts.values, y=col_counts.index, palette="viridis", ax=ax)
        ax.set_xlabel("Cantidad de valores no nulos")
        ax.set_ylabel("Columnas del DataFrame")
        ax.set_title("Conteo de registros por columna")
        st.pyplot(fig)
    with col4:
        st.subheader("Matriz de correlacion")
        cols_num = ['budget', 'revenue', 'popularity', 'vote_average', 'runtime']
        data_num = df[cols_num]
        fig, ax = plt.subplots()
        sns.heatmap(data_num.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    col5,col6 = st.columns(2)

    with col5:
        st.subheader("10 Géneros más rentables")
        genre_revenue = (
            df.groupby('main_genre')['revenue']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='revenue', y='main_genre', data=genre_revenue, ax=ax)
        ax.set_title("Promedio de ingresos por género", fontsize=12)
        ax.set_xlabel("Ingresos promedio")
        ax.set_ylabel("Género principal")
        st.pyplot(fig)
    with col6:
        st.subheader("10 Directores más rentables")
        genre_revenue = (
            df.groupby('director')['revenue']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x='revenue', y='director', data=genre_revenue, ax=ax)
        ax.set_title("Promedio de ingresos por director", fontsize=12)
        ax.set_xlabel("Ingresos promedio")
        ax.set_ylabel("Nombre")
        st.pyplot(fig)

    col7, = st.columns(1)
    with col7:
        st.subheader("Películas más taquilleras")

        # Ordenar todo el DataFrame por revenue
        df_sorted = df.sort_values(by='revenue', ascending=False).reset_index(drop=True)

        # Crear una columna que indique si está en el top 10
        df_sorted['is_top10'] = df_sorted.index < 10

        # Gráfica interactiva
        fig = px.bar(
            df_sorted.head(50),  # puedes cambiar a 100 si quieres ver más
            x='title_x',
            y='revenue',
            color='is_top10',
            color_discrete_map={True: 'gold', False: 'lightgray'},
            title="Ranking de películas más taquilleras (Top 50)",
            hover_data=['title_x', 'revenue'],
        )

        # Personalización
        fig.update_layout(
            xaxis_title="Película",
            yaxis_title="Ingresos ($)",
            showlegend=False,
            xaxis_tickangle=-45,
            template="plotly_dark",
        )

        st.plotly_chart(fig, use_container_width=True)


# ===================
# ========
# 🤖 Clasificación (Éxito/Fracaso)
# ===========================

if page == "👀 Comparativa":
    st.header("👀 Comparativa de modelos")

    # === Cargar JSON con resultados ===
    with open("model/metrics.json") as f:
        metrics_data = json.load(f)

    # Convertir a DataFrame
    df_metrics = pd.DataFrame(metrics_data)

    # Mostrar tabla
    st.subheader("📋 Resultados de validación")
    st.dataframe(df_metrics.style.format({
        "Best_Val_Acc": "{:.2%}",
        "Gap": "{:.2%}",
        "Min_Val_Loss": "{:.3f}"
    }))

    # === Selector de métrica para graficar ===
    metric_option = st.selectbox(
        "Selecciona la métrica a comparar:",
        ["Best_Val_Acc", "Gap", "Min_Val_Loss"]
    )

    # === Gráfica comparativa ===
    fig = px.bar(
        df_metrics,
        x="Técnica",
        y=metric_option,
        color="Técnica",
        text=df_metrics[metric_option].apply(lambda x: f"{x:.2f}"),
        title=f"📈 Comparación de {metric_option.replace('_', ' ')} por técnica",
        template="plotly_dark"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


if page == "🤖 Evaluador y predictor":
    st.header("🤖 Evaluador de modelos y recaudación")
    # === Seleccionar técnica ===
    selected_model = st.selectbox("🔍 Selecciona la técnica a evaluar:", df_metrics["Técnica"])
    selected_row = df_metrics[df_metrics["Técnica"] == selected_model].iloc[0]

    st.write(f"**Best Val Accuracy:** {selected_row['Best_Val_Acc']:.2%}")
    st.write(f"**Gap:** {selected_row['Gap']:.2%}")
    st.write(f"**Min Val Loss:** {selected_row['Min_Val_Loss']:.3f}")

    # === Mapeo de modelo ===
    model_map = {
        "Baseline": "baseline_model.h5",
        "Dropout": "dropout_model.h5",
        "L2": "l2_model.h5",
        "BatchNorm": "batchnorm_model.h5",
        "Combined": "combined_model.h5"
    }

    model_path = os.path.join("model", model_map[selected_model])

    # === Cargar modelos ===
    try:
        classifier = tf.keras.models.load_model(model_path)
        preprocessor = joblib.load("model/preprocessor.pkl")
        regressor = joblib.load("model/revenue_regressor.pkl")
        st.success(f"✅ Modelos '{selected_model}' y de regresión cargados correctamente.")
    except Exception as e:
        st.error(f"❌ Error al cargar modelos: {e}")
        st.stop()

    # === Formulario de entrada ===
    st.subheader("🎥 Ingresar datos de la película")

    budget = st.number_input("Presupuesto", min_value=0)
    popularity = st.number_input("Popularidad", min_value=0.0)
    vote_average = st.number_input("Calificación promedio", min_value=0.0, max_value=10.0)
    runtime = st.number_input("Duración (min)", min_value=0)
    main_genre = st.text_input("Género principal")
    director = st.text_input("Director")

    if st.button("📊 Predecir éxito y recaudación"):
        data = pd.DataFrame([{
            'budget': budget,
            'popularity': popularity,
            'vote_average': vote_average,
            'runtime': runtime,
            'main_genre': main_genre,
            'director': director
        }])

        # Transformar entrada
        X_input = preprocessor.transform(data)

        # Predicción de éxito (clasificación)
        success_prob = classifier.predict(X_input)[0][0]

        # Predicción de recaudación (regresión)
        revenue_pred = regressor.predict(data)[0]

        # Mostrar resultados
        st.metric("Probabilidad de éxito", f"{success_prob*100:.2f}%")
        st.metric("Recaudación estimada", f"${revenue_pred:,.0f}")


        st.markdown("""---""")
        with open("model/regression_metrics.json") as f:
            regression_metrics = json.load(f)

        # Mostrar métricas del modelo de regresión
        st.subheader("📈 Desempeño del modelo de recaudación")
        col1, col2 = st.columns(2)
        col1.metric("R² (Coeficiente de determinación)", f"{regression_metrics['R2']:.3f}")
        col2.metric("RMSE (Error cuadrático medio)", f"${regression_metrics['RMSE']:,.0f}")

            # === Explicación ===
        with st.expander("ℹ️ ¿Qué significan estas métricas?"):
            st.markdown("""
            - **R² (Coeficiente de determinación)**: Indica qué tan bien el modelo explica la variabilidad de la recaudación.  
            Un valor de **1.0** significa una predicción perfecta, y valores cercanos a **0** indican baja precisión.
            
            - **RMSE (Raíz del Error Cuadrático Medio)**: Mide el error promedio de las predicciones del modelo.  
            Cuanto **menor sea el RMSE**, más precisas son las predicciones.  
            Se expresa en las mismas unidades que la variable objetivo (dólares 💵).
            """)