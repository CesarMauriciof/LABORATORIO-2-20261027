# Laboratorio 2 · Predicción de aprobación con un MLP

Aplicación web que predice si un estudiante aprobará una asignatura a partir de sus
calificaciones, asistencia y hábitos. El modelo es un **perceptrón multicapa (MLP)**
entrenado con `scikit-learn` sobre el dataset `dataset_estudiantes_balanceado.csv`
(600 registros, 300 aprobados y 300 reprobados).

## Resultados del modelo

| Métrica | Valor |
| --- | --- |
| Accuracy | 0.9667 |
| Precision | 0.9828 |
| Recall | 0.9500 |
| F1-Score | 0.9661 |
| ROC-AUC | 0.9969 |

- Arquitectura: `6 → 32 → 16 → 1` (ReLU, Adam, early stopping).
- División: 480 registros de entrenamiento y 120 de prueba (estratificada, semilla 42).
- F1 promedio en validación cruzada de 5 folds: **0.9857**.
- Matriz de confusión en prueba: `[[59, 1], [3, 57]]`.

El análisis completo y las conclusiones están en [`reports/analisis.md`](reports/analisis.md).
El análisis exploratorio está en [`reports/eda_resumen.md`](reports/eda_resumen.md).

## Estructura del proyecto

```
├── app.py                            # Aplicación Streamlit (entrypoint)
├── MLP_estudiantes.ipynb             # Notebook con el análisis y el entrenamiento paso a paso
├── dataset_estudiantes_balanceado.csv
├── models/
│   └── mlp_estudiantes.joblib        # Pipeline entrenado (imputador + escalador + MLP)
├── reports/
│   ├── analisis.md                   # Conclusiones del problema y de las métricas
│   ├── curva_perdida.png
│   ├── matriz_confusion.png
│   ├── eda_*.png
│   ├── eda_resumen.md
│   ├── estadisticas_descriptivas.csv
│   ├── metrics.json
│   └── reporte_clasificacion.txt
├── scripts/
│   ├── analisis_exploratorio.py      # EDA y figuras
│   └── entrenar_modelo.py            # Entrenamiento, búsqueda de hiperparámetros y evaluación
├── .streamlit/config.toml            # Tema de la app
├── requirements.txt
└── README.md
```

## Cómo se construyó el modelo

1. **Carga e imputación**: `SimpleImputer(strategy="median")` por columna. El dataset no
   tiene nulos, pero el paso queda dentro del pipeline para que la app tolere datos
   incompletos en producción.
2. **Balanceo**: se verifica la proporción 50/50; si estuviera desbalanceado, se aplica
   submuestreo de la clase mayoritaria antes de dividir.
3. **Escalado**: `StandardScaler` (media 0, varianza 1) dentro del pipeline, ajustado
   únicamente con el set de entrenamiento.
4. **MLP**: `MLPClassifier` con activación ReLU, optimizador Adam y *early stopping*.
   La arquitectura y la regularización se eligen con `GridSearchCV` (5 folds):
   capas ocultas `(16,)`, `(32,)`, `(32,16)`, `(64,32)`; `alpha` en `[1e-4, 1e-3, 1e-2]`
   y tasa de aprendizaje en `[0.001, 0.01]`.
5. **Evaluación**: una sola medición sobre el 20 % de prueba, con matriz de confusión,
   precision, recall, F1, accuracy y ROC-AUC por clase.

`Calificacion_Final` **no se usa como entrada**: es una salida del dataset y el objetivo
`Aprobado` se deriva de ella, así que incluirla produciría fuga de información.

## Ejecutar en local

Requiere **Python 3.12 o superior**.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# 1) Análisis exploratorio (opcional)
python scripts/analisis_exploratorio.py

# 2) Entrenamiento y métricas (regenera models/ y reports/)
python scripts/entrenar_modelo.py

# 3) Aplicación web
python -m streamlit run app.py
```

La app abre en `http://localhost:8501`. Si `streamlit` está en el `PATH`, también puedes
usar `streamlit run app.py`.

La pestaña **Modelo y métricas** incluye un botón para reentrenar desde la propia app;
tarda alrededor de 30 segundos y actualiza el modelo y los reportes.

## Notebook de Jupyter

`MLP_estudiantes.ipynb` contiene el mismo flujo explicado paso a paso: carga y calidad de
los datos, EDA, imputación por medianas, estandarización, búsqueda de arquitectura,
evaluación con matriz de confusión, importancia de variables y conclusiones. Ya está
ejecutado, así que puedes ver los resultados sin correrlo; para abrirlo:

```bash
pip install jupyterlab
jupyter lab MLP_estudiantes.ipynb
```

El notebook tarda alrededor de 90 segundos en ejecutarse completo porque repite la
búsqueda de hiperparámetros. No sobrescribe el modelo de la app; para eso usa
`python scripts/entrenar_modelo.py`.


## Despliegue en Streamlit Community Cloud (endpoint público)

El repositorio ya está listo para subirse: incluye `requirements.txt`, el modelo entrenado
y los reportes. No se necesitan secretos ni variables de entorno.

### 1. Subir el proyecto a GitHub

Crea un repositorio vacío en GitHub (por ejemplo `mlp-aprobacion-estudiantes`) y ejecuta
en esta carpeta:

```bash
git init
git add .
git commit -m "Laboratorio 2: MLP para predicción de aprobación"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/TU-REPOSITORIO.git
git push -u origin main
```

### 2. Crear la app en Streamlit Cloud

1. Entra a <https://share.streamlit.io> e inicia sesión con GitHub.
2. Selecciona **Create app** (o **New app**).
3. Elige el repositorio, la rama `main` y el archivo principal `app.py`.
4. En **Advanced settings**, selecciona **Python 3.13** (o la versión más reciente
   disponible, siempre que sea 3.12 o superior).
5. Presiona **Deploy** y espera a que termine la instalación.

### 3. Registrar el endpoint

Cuando la app esté publicada, Streamlit te dará una URL con esta forma:

```
https://TU-USUARIO-TU-APP.streamlit.app
```

Pega aquí tu enlace final:

**Endpoint:** _pendiente de publicar_

> Streamlit publica una aplicación web interactiva. Esa URL es el endpoint del servicio
> (formulario + predicción en pantalla). No expone una API REST con JSON; para eso haría
> falta desplegar un servicio aparte (por ejemplo FastAPI en Render o Railway).

## Uso de la aplicación

- **Predicción**: ajusta Tareas, Exámenes, Asistencia, horas de estudio, horas de sueño y
  participación extracurricular. Al enviar el formulario se muestra la clase estimada
  (Aprobado/Reprobado), la probabilidad y una barra de confianza.
- **Modelo y métricas**: matriz de confusión, métricas globales y por clase, arquitectura
  elegida, curva de pérdida y botón de reentrenamiento.

## Limitaciones

El dataset es sintético y el objetivo se generó a partir de una combinación lineal de las
variables académicas. Por eso el problema es casi linealmente separable y el rendimiento
ronda el 97 %, con probabilidades muy cercanas a 0 o 1. En un escenario real harían falta
datos más ricos, validación temporal y revisión de sesgos. `Sueno_Horas` y
`Extracurriculares` aportan muy poca señal. La predicción debe usarse como apoyo
pedagógico, nunca como decisión automática sobre un estudiante.

## Solución de problemas

- **`ModuleNotFoundError`**: activa el entorno virtual e instala `requirements.txt`.
- **La app no encuentra el modelo**: ejecuta `python scripts/entrenar_modelo.py` o usa el
  botón de reentrenamiento.
- **Advertencia de versión de scikit-learn al cargar el modelo**: las versiones de
  `requirements.txt` deben coincidir con las usadas al entrenar. Si cambias de versión,
  reentrena el modelo.
- **El puerto 8501 está ocupado**: `python -m streamlit run app.py --server.port 8502`.
# LABORATORIO-2-20261027
