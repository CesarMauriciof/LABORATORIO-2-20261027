"""Aplicación web del Laboratorio 2: predicción de aprobación con un MLP.

Ejecutar en local:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from scripts.entrenar_modelo import (
    CARACTERISTICAS,
    RUTA_METRICAS,
    RUTA_MODELO,
    RUTA_REPORTES,
    entrenar_modelo,
)

RAIZ = Path(__file__).resolve().parent
RUTA_MATRIZ = RUTA_REPORTES / "matriz_confusion.png"
RUTA_CURVA = RUTA_REPORTES / "curva_perdida.png"

st.set_page_config(
    page_title="Predicción de aprobación · MLP",
    page_icon=":material/school:",
    layout="wide",
)

st.session_state.setdefault("resultado_prediccion", None)


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    """Carga el pipeline imputador + escalador + MLP desde disco."""
    if not RUTA_MODELO.exists():
        return None
    return joblib.load(RUTA_MODELO)


def leer_metricas() -> dict | None:
    if not RUTA_METRICAS.exists():
        return None
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))


def tabla_metricas(metricas: dict) -> pd.DataFrame:
    m = metricas["metricas_prueba"]
    return pd.DataFrame(
        {
            "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Valor": [
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['roc_auc']:.4f}",
            ],
        }
    )


def tabla_por_clase(metricas: dict) -> pd.DataFrame:
    reporte = metricas["reporte_clasificacion"]
    filas = []
    for etiqueta in metricas["matriz_confusion"]["etiquetas"]:
        datos = reporte[etiqueta]
        filas.append(
            {
                "Clase": etiqueta,
                "Precision": f"{datos['precision']:.3f}",
                "Recall": f"{datos['recall']:.3f}",
                "F1-Score": f"{datos['f1-score']:.3f}",
                "Casos": int(datos["support"]),
            }
        )
    return pd.DataFrame(filas)


metricas = leer_metricas()

st.title("Predicción de aprobación con un MLP")
st.caption(
    "Laboratorio 2 · Perceptrón multicapa entrenado con scikit-learn sobre "
    "600 registros académicos y de hábitos de estudiantes."
)

mensaje = st.session_state.pop("mensaje_entrenamiento", None)
if mensaje:
    st.success(mensaje, icon=":material/check_circle:")

if metricas:
    m = metricas["metricas_prueba"]
    columnas = st.columns(4, gap="medium")
    columnas[0].metric(
        "Accuracy",
        f"{m['accuracy']:.2%}",
        border=True,
        help="Proporción de aciertos sobre el set de prueba.",
    )
    columnas[1].metric(
        "F1-Score",
        f"{m['f1']:.4f}",
        border=True,
        help="Media armónica entre precision y recall.",
    )
    columnas[2].metric(
        "Recall",
        f"{m['recall']:.2%}",
        border=True,
        help="Porcentaje de estudiantes aprobados que el modelo detecta.",
    )
    columnas[3].metric(
        "ROC-AUC",
        f"{m['roc_auc']:.4f}",
        border=True,
        help="Capacidad de separar ambas clases en todos los umbrales.",
    )
    st.caption(
        f"Evaluado sobre {metricas['split']['prueba']} estudiantes que no se usaron "
        f"durante el entrenamiento · Arquitectura "
        f"{' → '.join(str(n) for n in metricas['arquitectura']['capas_ocultas'])}."
    )

tab_prediccion, tab_metricas = st.tabs(
    ["Predicción", "Modelo y métricas"], key="pestanas"
)

with tab_prediccion:
    st.caption(
        "Completa las seis variables y presiona **Predecir aprobación**. "
        "Los valores iniciales corresponden a un estudiante promedio del dataset."
    )

    columna_formulario, columna_resultado = st.columns([1.35, 1], gap="large")

    with columna_formulario:
        with st.container(border=True):
            with st.form("formulario_prediccion", border=False):
                rendimiento, habitos = st.columns(2, gap="large")

                with rendimiento:
                    st.markdown("**Rendimiento académico**")
                    tareas = st.slider(
                        "Tareas (0-100)",
                        min_value=0.0,
                        max_value=100.0,
                        value=66.0,
                        step=0.5,
                        help="Calificación acumulada de los entregables prácticos.",
                        key="tareas",
                    )
                    examenes = st.slider(
                        "Exámenes (0-100)",
                        min_value=0.0,
                        max_value=100.0,
                        value=60.0,
                        step=0.5,
                        help="Calificación del examen parcial y final.",
                        key="examenes",
                    )
                    asistencia = st.slider(
                        "Asistencia (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=70.0,
                        step=0.5,
                        help="Porcentaje de asistencia a clases.",
                        key="asistencia",
                    )

                with habitos:
                    st.markdown("**Hábitos y entorno**")
                    estudio = st.slider(
                        "Horas de estudio por semana",
                        min_value=0.0,
                        max_value=30.0,
                        value=13.5,
                        step=0.5,
                        help="Promedio de horas de estudio independiente por semana.",
                        key="estudio_horas",
                    )
                    sueno = st.slider(
                        "Horas de sueño por noche",
                        min_value=0.0,
                        max_value=12.0,
                        value=6.5,
                        step=0.5,
                        help="Promedio de horas diarias de sueño.",
                        key="sueno_horas",
                    )
                    extracurriculares = st.toggle(
                        "Participa en actividades extracurriculares",
                        value=False,
                        help="Talleres o deportes (1 = sí, 0 = no).",
                        key="extracurriculares",
                    )

                enviado = st.form_submit_button(
                    "Predecir aprobación",
                    type="primary",
                    icon=":material/online_prediction:",
                    width="stretch",
                    key="enviar_prediccion",
                )

    with columna_resultado:
        with st.container(border=True):
            st.subheader("Resultado")

            if enviado:
                modelo = cargar_modelo()
                if modelo is None:
                    st.session_state["resultado_prediccion"] = None
                    st.error(
                        "No se encontró el modelo entrenado. Ve a **Modelo y métricas** "
                        "y usa el botón de reentrenamiento.",
                        icon=":material/error:",
                    )
                else:
                    entrada = pd.DataFrame(
                        [
                            {
                                "Tareas": tareas,
                                "Examenes": examenes,
                                "Asistencia": asistencia,
                                "Estudio_Horas": estudio,
                                "Sueno_Horas": sueno,
                                "Extracurriculares": int(extracurriculares),
                            }
                        ]
                    )[CARACTERISTICAS]

                    try:
                        probabilidad = float(modelo.predict_proba(entrada)[0, 1])
                        st.session_state["resultado_prediccion"] = {
                            "probabilidad": probabilidad,
                            "aprobado": probabilidad >= 0.5,
                        }
                    except Exception as exc:  # pragma: no cover - defensivo
                        st.error(
                            f"No se pudo generar la predicción: {exc}",
                            icon=":material/error:",
                        )

            resultado = st.session_state["resultado_prediccion"]
            if resultado is None:
                st.info(
                    "Aún no hay una predicción. Ajusta las variables y envía el formulario.",
                    icon=":material/query_stats:",
                )
            else:
                probabilidad = resultado["probabilidad"]
                aprobado = resultado["aprobado"]

                if aprobado:
                    st.success(
                        "El modelo estima que el estudiante **aprobará** la asignatura.",
                        icon=":material/check_circle:",
                        title="Aprobado",
                    )
                else:
                    st.error(
                        "El modelo estima que el estudiante **no aprobará** la asignatura.",
                        icon=":material/cancel:",
                        title="Reprobado",
                    )

                st.metric(
                    "Probabilidad de aprobación",
                    f"{probabilidad:.1%}",
                    border=True,
                )
                st.progress(
                    probabilidad,
                    text="Probabilidad estimada de aprobar",
                )
                st.caption(
                    "Decisión con umbral 0,50. Una probabilidad cercana al 50 % "
                    "indica un caso límite: conviene revisarlo con otras fuentes."
                )

            st.caption(
                "Esta predicción es una estimación estadística y no debe usarse "
                "como decisión automática sobre un estudiante."
            )

with tab_metricas:
    if not metricas:
        st.warning(
            "Todavía no hay métricas. Entrena el modelo para generarlas.",
            icon=":material/warning:",
        )
    else:
        st.subheader("Desempeño sobre el set de prueba")
        st.caption(
            f"División estratificada {metricas['split']['entrenamiento']}/"
            f"{metricas['split']['prueba']} · semilla {metricas['split']['semilla']} · "
            f"F1 en validación cruzada (5 folds): "
            f"{metricas['validacion_cruzada']['f1_promedio']:.4f}."
        )

        columna_matriz, columna_tablas = st.columns([1, 1.1], gap="large")

        with columna_matriz:
            if RUTA_MATRIZ.exists():
                st.image(
                    str(RUTA_MATRIZ),
                    caption="Matriz de confusión sobre el set de prueba",
                    alt=(
                        "Matriz de confusión con "
                        f"{metricas['matriz_confusion']['verdaderos_positivos']} verdaderos positivos, "
                        f"{metricas['matriz_confusion']['verdaderos_negativos']} verdaderos negativos, "
                        f"{metricas['matriz_confusion']['falsos_positivos']} falsos positivos y "
                        f"{metricas['matriz_confusion']['falsos_negativos']} falsos negativos."
                    ),
                    width="stretch",
                )

        with columna_tablas:
            st.markdown("**Métricas globales**")
            st.dataframe(
                tabla_metricas(metricas),
                hide_index=True,
                alt="Tabla con accuracy, precision, recall, F1 y ROC-AUC del modelo.",
            )
            st.markdown("**Detalle por clase**")
            st.dataframe(
                tabla_por_clase(metricas),
                hide_index=True,
                alt="Tabla con precision, recall, F1 y cantidad de casos por clase.",
            )

        st.divider()

        with st.expander(
            "Arquitectura, hiperparámetros y entrenamiento",
            icon=":material/tune:",
        ):
            parametros = metricas["mejores_parametros"]
            arquitectura = metricas["arquitectura"]

            resumen = st.columns(4, gap="medium")
            resumen[0].metric(
                "Capas ocultas",
                " → ".join(str(n) for n in arquitectura["capas_ocultas"]),
                border=True,
            )
            resumen[1].metric(
                "Neuronas de entrada", len(CARACTERISTICAS), border=True
            )
            resumen[2].metric("Alpha", parametros["mlp__alpha"], border=True)
            resumen[3].metric(
                "Tasa de aprendizaje",
                parametros["mlp__learning_rate_init"],
                border=True,
            )

            st.markdown(
                f"- Activación: **{arquitectura['activacion'].upper()}** · "
                f"optimizador: **{arquitectura['solver']}** · "
                f"épocas entrenadas: **{arquitectura['epocas_entrenadas']}**."
            )
            st.markdown(
                "- Pipeline: `SimpleImputer(median)` → `StandardScaler` → `MLPClassifier`."
            )
            st.markdown(
                f"- Dataset: **{metricas['dataset']['registros_usados']}** registros "
                f"({metricas['dataset']['aprobados']} aprobados / "
                f"{metricas['dataset']['reprobados']} reprobados)."
            )
            if RUTA_CURVA.exists():
                st.image(
                    str(RUTA_CURVA),
                    caption="Curva de pérdida del modelo final",
                    alt="Gráfico de la pérdida logarítmica por época durante el entrenamiento.",
                    width="stretch",
                )

        st.info(
            "`Calificacion_Final` se excluyó de las entradas porque es una salida del "
            "dataset y el objetivo `Aprobado` se deriva de ella. Usarla produciría fuga "
            "de información y métricas artificialmente altas.",
            icon=":material/info:",
            title="Sobre las variables de entrada",
        )

        st.divider()
        st.subheader("Reentrenar el modelo")
        st.caption(
            "Vuelve a ejecutar la imputación, el escalado y la búsqueda de capas ocultas "
            "con el dataset actual. Tarda alrededor de un minuto."
        )

        if st.button(
            "Reentrenar modelo",
            icon=":material/model_training:",
            type="primary",
            key="reentrenar",
        ):
            try:
                with st.spinner("Entrenando y evaluando el MLP…"):
                    nuevas_metricas = entrenar_modelo(verbose=False)
                cargar_modelo.clear()
                st.session_state["mensaje_entrenamiento"] = (
                    "Modelo reentrenado. Accuracy en el set de prueba: "
                    f"{nuevas_metricas['metricas_prueba']['accuracy']:.2%}."
                )
                st.rerun()
            except Exception as exc:  # pragma: no cover - defensivo
                st.error(
                    f"No se pudo reentrenar el modelo: {exc}",
                    icon=":material/error:",
                )

st.divider()
st.caption(
    "Laboratorio 2 · MLP para analítica académica · Los datos son sintéticos y "
    "se usan con fines educativos."
)
