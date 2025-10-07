import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el dataset
st.title('Boston Housing - Análisis y Regresión')

DATA_PATH = 'housing.csv'

def cargar_datos():
    try:
        df = pd.read_csv(DATA_PATH)
        # Identificar columnas categóricas y aplicar One-Hot Encoding
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        return df
    except Exception as e:
        st.error(f'Error al cargar el archivo: {e}')
        return None

df = cargar_datos()
if df is not None:
    st.subheader('Primeras filas del dataset')
    st.dataframe(df.head())

    st.subheader('Estadísticas descriptivas')
    st.dataframe(df.describe())

    st.subheader('Visualización de variables')
    columnas = st.multiselect('Selecciona variables para visualizar', df.columns.tolist(), default=df.columns[:2].tolist())
    if columnas:
        fig, ax = plt.subplots()
        sns.pairplot(df[columnas])
        st.pyplot(fig)

    st.subheader('Correlación entre variables')
    fig_corr, ax_corr = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader('Modelos de Regresión')
    # Solo columnas numéricas como objetivo
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    target = st.selectbox('Selecciona la variable objetivo', numeric_cols)
    features = st.multiselect('Selecciona variables predictoras', [col for col in df.columns if col != target], default=[col for col in df.columns if col != target][:2])

    if target and features:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        st.write('Coeficientes:', dict(zip(features, modelo.coef_)))
        st.write('Intercepto:', modelo.intercept_)
        st.write('MSE:', mean_squared_error(y_test, y_pred))
        st.write('R2:', r2_score(y_test, y_pred))
        st.line_chart(pd.DataFrame({'Real': y_test, 'Predicción': y_pred}).reset_index(drop=True))
else:
    st.warning('No se pudo cargar el dataset. Asegúrate de que housing.csv está en la carpeta del proyecto.')
