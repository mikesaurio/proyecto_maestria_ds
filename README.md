# 🎬 Predicción de Éxito y Recaudación de Películas

Este proyecto permite **evaluar el éxito comercial de películas** y **predecir su recaudación** a partir de características como presupuesto, popularidad, duración, calificación, género y director.  
Utiliza **redes neuronales con TensorFlow** para clasificación y **Random Forest** para regresión.

---

## 🚀 Características principales

- Interfaz interactiva con **Streamlit**.  
- Modelos de clasificación entrenados con distintas técnicas:
  - **Baseline**
  - **Dropout**
  - **L2 Regularization**
  - **Batch Normalization**
  - **Combined (Dropout + L2 + BatchNorm)**
- Modelo de **regresión RandomForest** para predecir la recaudación.
- Visualización de métricas como **Accuracy, R² y RMSE**.
- Selección dinámica del modelo desde la interfaz.

---

## 🧱 Estructura del proyecto

    ├── datalake/
    │ └── bronze/ # Archivos originales descargados
    │ └── gold/ # Data limpia
    ├── model/
    │ ├── baseline_model.h5
    │ ├── dropout_model.h5
    │ ├── l2_model.h5
    │ ├── batchnorm_model.h5
    │ ├── combined_model.h5
    │ ├── preprocessor.pkl
    │ ├── revenue_regressor.pkl
    │ ├── metrics.json
    │ └── regression_metrics.json
    ├── app.py # Aplicación principal de Streamlit
    ├── requirements.txt # Dependencias del proyecto
    └── README.md # Este archivo



---

## 🧰 Requisitos previos

Asegúrate de tener **Python 3.9 o superior** instalado.

### Crear y activar un entorno virtual

En Windows:
```bash
py -3.10 -m venv venv
 .\venv\Scripts\activate
```

En macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
### Instalar dependencias

Instala las librerías del proyecto desde el archivo requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 🧠 Modelos utilizados

🔹 Clasificación de Éxito

Entrenamos 5 modelos distintos basados en redes neuronales densas para predecir si una película será exitosa (success = 1) o no (success = 0).

Cada modelo aplica una técnica diferente:

| Modelo        | Técnica Principal                 | Objetivo                      |
| ------------- | --------------------------------- | ----------------------------- |
| **Baseline**  | Red neuronal simple               | Punto de comparación          |
| **Dropout**   | Evitar sobreajuste aleatoriamente | Generalización                |
| **L2**        | Penalización de pesos grandes     | Regularización                |
| **BatchNorm** | Normalización entre capas         | Estabilidad del entrenamiento |
| **Combined**  | Dropout + L2 + BatchNorm          | Mayor robustez                |



🔹 Clasificación de Éxito

    Se usó un Random Forest Regressor para predecir la recaudación estimada (revenue).

🔹 Métricas evaluadas:

    R² (Coeficiente de determinación): mide qué tan bien el modelo explica la variabilidad de la recaudación.

    RMSE (Raíz del Error Cuadrático Medio): mide el error promedio de las predicciones en dólares.

---
## 🖥️ Ejecutar la aplicación

```bash
streamlit run app.py
```
---

##  🧑‍💻Contacto
```bash
Miguel Moran @yosoymikesaurio
```