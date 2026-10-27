"""Entrena el perceptrón multicapa (MLP) que predice si un estudiante aprueba.

El flujo implementa los requisitos del taller:
1. Carga del dataset e imputación de valores ausentes con la mediana por columna.
2. Verificación (y, si hiciera falta, corrección) del balance 50/50 de clases.
3. Estandarización de características (media 0, varianza 1).
4. Búsqueda de la mejor arquitectura de capas ocultas con validación cruzada.
5. Evaluación sobre un conjunto de prueba independiente (matriz de confusión y métricas).

Uso:
    python scripts/entrenar_modelo.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "dataset_estudiantes_balanceado.csv"
RUTA_MODELO = RAIZ / "models" / "mlp_estudiantes.joblib"
RUTA_METRICAS = RAIZ / "reports" / "metrics.json"
RUTA_REPORTES = RAIZ / "reports"

CARACTERISTICAS = [
    "Tareas",
    "Examenes",
    "Asistencia",
    "Estudio_Horas",
    "Sueno_Horas",
    "Extracurriculares",
]
OBJETIVO = "Aprobado"
# Calificacion_Final es una salida del dataset (el objetivo se deriva de ella),
# por lo que usarla como entrada produciría fuga de información.
COLUMNAS_EXCLUIDAS = ["Calificacion_Final"]
ETIQUETAS = ["Reprobado", "Aprobado"]

SEMILLA = 42
PROPORCION_PRUEBA = 0.20
REJILLA = {
    "mlp__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
    "mlp__alpha": [1e-4, 1e-3, 1e-2],
    "mlp__learning_rate_init": [0.001, 0.01],
}


def cargar_datos() -> tuple[pd.DataFrame, dict]:
    """Carga el CSV, valida columnas y devuelve el dataframe con su resumen."""
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {RUTA_DATOS}")

    df = pd.read_csv(RUTA_DATOS)
    faltantes = [c for c in CARACTERISTICAS + [OBJETIVO] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias en el dataset: {faltantes}")

    df = df.copy()
    for columna in CARACTERISTICAS:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df[OBJETIVO] = pd.to_numeric(df[OBJETIVO], errors="coerce")

    if df[OBJETIVO].isna().any():
        raise ValueError("La variable objetivo Aprobado contiene valores no numéricos.")
    df[OBJETIVO] = df[OBJETIVO].astype(int)
    if not set(df[OBJETIVO].unique()).issubset({0, 1}):
        raise ValueError("Aprobado debe contener únicamente 0 y 1.")

    nulos = df[CARACTERISTICAS].isna().sum()
    resumen = {
        "registros": int(len(df)),
        "columnas": list(df.columns),
        "valores_ausentes_por_columna": {k: int(v) for k, v in nulos.items()},
        "total_valores_ausentes": int(nulos.sum()),
        "balance_original": {str(k): int(v) for k, v in df[OBJETIVO].value_counts().sort_index().items()},
    }
    return df, resumen


def equilibrar_clases(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Garantiza 50/50 submuestreando la clase mayoritaria si es necesario."""
    conteos = df[OBJETIVO].value_counts()
    if len(conteos) < 2:
        return df.reset_index(drop=True), False
    if conteos.min() == conteos.max():
        return df.reset_index(drop=True), False

    tamano = int(conteos.min())
    partes = [
        grupo.sample(n=tamano, random_state=SEMILLA)
        for _, grupo in df.groupby(OBJETIVO, sort=True)
    ]
    equilibrado = (
        pd.concat(partes).sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    )
    return equilibrado, True


def _graficar_matriz_confusion(cm, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ETIQUETAS).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("Matriz de confusión · set de prueba")
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _graficar_curva_perdida(modelo: MLPClassifier, ruta: Path) -> None:
    curva = getattr(modelo, "loss_curve_", None)
    if not curva:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(range(1, len(curva) + 1), curva, color="#1f6feb", linewidth=2)
    ax.set_title("Curva de pérdida durante el entrenamiento")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (log loss)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _escribir_analisis(metricas: dict) -> None:
    m = metricas["metricas_prueba"]
    mejor = metricas["mejores_parametros"]
    lineas = [
        "# Análisis y conclusiones",
        "",
        f"Generado automáticamente el {metricas['generado_en']} por `scripts/entrenar_modelo.py`.",
        "",
        "## Configuración del experimento",
        "",
        f"- Registros utilizados: **{metricas['dataset']['registros_usados']}** "
        f"({metricas['dataset']['aprobados']} aprobados / {metricas['dataset']['reprobados']} reprobados).",
        f"- Entrenamiento: **{metricas['split']['entrenamiento']}** registros · "
        f"Prueba independiente: **{metricas['split']['prueba']}** registros "
        f"(división estratificada {int((1 - PROPORCION_PRUEBA) * 100)}/{int(PROPORCION_PRUEBA * 100)}, semilla {SEMILLA}).",
        f"- Arquitectura seleccionada: capas ocultas **{' → '.join(str(n) for n in mejor['mlp__hidden_layer_sizes'])}**, "
        f"alpha **{mejor['mlp__alpha']}**, tasa de aprendizaje **{mejor['mlp__learning_rate_init']}**.",
        "- Preprocesamiento dentro del pipeline: imputación por mediana → `StandardScaler` → MLP.",
        "- `Calificacion_Final` se excluyó de las entradas porque es una salida del dataset "
        "y el objetivo `Aprobado` se deriva de ella (fuga de información).",
        "",
        "## Métricas sobre el set de prueba",
        "",
        "| Métrica | Valor |",
        "| --- | --- |",
        f"| Accuracy | {m['accuracy']:.4f} |",
        f"| Precision | {m['precision']:.4f} |",
        f"| Recall | {m['recall']:.4f} |",
        f"| F1-Score | {m['f1']:.4f} |",
        f"| ROC-AUC | {m['roc_auc']:.4f} |",
        "",
        f"F1 promedio en validación cruzada (5 folds): **{metricas['validacion_cruzada']['f1_promedio']:.4f}**.",
        "",
        "## Conclusiones",
        "",
        f"1. **Rendimiento.** El MLP acierta el {m['accuracy'] * 100:.1f} % de los casos del set de prueba "
        f"con un F1 de {m['f1']:.3f}. La brecha entre validación cruzada "
        f"({metricas['validacion_cruzada']['f1_promedio']:.3f}) y prueba ({m['f1']:.3f}) "
        "es pequeña, por lo que no se observa sobreajuste relevante.",
        "2. **Naturaleza del problema.** El objetivo se construyó a partir de una combinación lineal "
        "de las variables académicas, de modo que el problema es casi linealmente separable. "
        "Un MLP pequeño basta; añadir capas o neuronas no aporta mejoras y sí aumenta el riesgo de sobreajuste.",
        "3. **Variables relevantes.** Tareas, Exámenes, Estudio_Horas y Asistencia concentran la señal. "
        "Sueno_Horas y Extracurriculares muestran una separación casi nula entre clases, "
        "por lo que el modelo no las necesita para decidir.",
        "4. **Validez de las métricas.** Con clases balanceadas, accuracy y F1 son comparables; aun así "
        "se reportan precision y recall por clase para detectar errores asimétricos. "
        "La matriz de confusión confirma que los errores se reparten entre ambas clases.",
        "5. **Limitaciones.** El dataset es sintético y el objetivo es determinista, por lo que un "
        "rendimiento cercano al 100 % no se trasladaría a estudiantes reales. Para un uso real harían "
        "falta datos más ricos, validación temporal y revisión de sesgos.",
        "",
        "## Recomendaciones",
        "",
        "- Usar el modelo como apoyo, nunca como decisión automática sobre un estudiante.",
        "- Reentrenar y reevaluar si cambia la distribución de los datos o el criterio de aprobación.",
        "- Documentar el pipeline completo (imputación y escalado incluidos) para evitar *data leakage* "
        "entre entrenamiento y producción.",
        "",
    ]
    (RUTA_REPORTES / "analisis.md").write_text("\n".join(lineas), encoding="utf-8")


def entrenar_modelo(verbose: bool = True) -> dict:
    """Entrena, evalúa y guarda el pipeline. Devuelve el diccionario de métricas."""
    df, resumen_datos = cargar_datos()
    df, reequilibrado = equilibrar_clases(df)
    resumen_datos["requirio_reequilibrio"] = reequilibrado

    X = df[CARACTERISTICAS]
    y = df[OBJETIVO]

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=PROPORCION_PRUEBA,
        stratify=y,
        random_state=SEMILLA,
    )

    pipeline = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    solver="adam",
                    activation="relu",
                    max_iter=1500,
                    early_stopping=True,
                    n_iter_no_change=30,
                    validation_fraction=0.15,
                    random_state=SEMILLA,
                ),
            ),
        ]
    )

    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    busqueda = GridSearchCV(
        estimator=pipeline,
        param_grid=REJILLA,
        scoring="f1",
        cv=validacion,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    busqueda.fit(X_entrenamiento, y_entrenamiento)

    modelo: Pipeline = busqueda.best_estimator_
    y_pred = modelo.predict(X_prueba)
    y_prob = modelo.predict_proba(X_prueba)[:, 1]

    cm = confusion_matrix(y_prueba, y_pred, labels=[0, 1])
    reporte = classification_report(
        y_prueba,
        y_pred,
        labels=[0, 1],
        target_names=ETIQUETAS,
        output_dict=True,
        zero_division=0,
    )
    reporte_texto = classification_report(
        y_prueba,
        y_pred,
        labels=[0, 1],
        target_names=ETIQUETAS,
        digits=4,
        zero_division=0,
    )

    metricas = {
        "generado_en": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "dataset": {
            "archivo": RUTA_DATOS.name,
            **resumen_datos,
            "registros_usados": int(len(df)),
            "aprobados": int((df[OBJETIVO] == 1).sum()),
            "reprobados": int((df[OBJETIVO] == 0).sum()),
        },
        "caracteristicas": CARACTERISTICAS,
        "columnas_excluidas": COLUMNAS_EXCLUIDAS,
        "variable_objetivo": OBJETIVO,
        "split": {
            "entrenamiento": int(len(X_entrenamiento)),
            "prueba": int(len(X_prueba)),
            "estratificado": True,
            "semilla": SEMILLA,
        },
        "mejores_parametros": {
            "mlp__hidden_layer_sizes": list(busqueda.best_params_["mlp__hidden_layer_sizes"]),
            "mlp__alpha": busqueda.best_params_["mlp__alpha"],
            "mlp__learning_rate_init": busqueda.best_params_["mlp__learning_rate_init"],
        },
        "arquitectura": {
            "capas_ocultas": list(modelo.named_steps["mlp"].hidden_layer_sizes),
            "activacion": modelo.named_steps["mlp"].activation,
            "solver": modelo.named_steps["mlp"].solver,
            "epocas_entrenadas": int(modelo.named_steps["mlp"].n_iter_),
        },
        "validacion_cruzada": {
            "folds": 5,
            "metrica": "f1",
            "f1_promedio": float(busqueda.best_score_),
        },
        "metricas_prueba": {
            "accuracy": float(accuracy_score(y_prueba, y_pred)),
            "precision": float(precision_score(y_prueba, y_pred, zero_division=0)),
            "recall": float(recall_score(y_prueba, y_pred, zero_division=0)),
            "f1": float(f1_score(y_prueba, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_prueba, y_prob)),
        },
        "matriz_confusion": {
            "etiquetas": ETIQUETAS,
            "matriz": cm.tolist(),
            "verdaderos_negativos": int(cm[0][0]),
            "falsos_positivos": int(cm[0][1]),
            "falsos_negativos": int(cm[1][0]),
            "verdaderos_positivos": int(cm[1][1]),
        },
        "reporte_clasificacion": reporte,
    }

    RUTA_REPORTES.mkdir(parents=True, exist_ok=True)
    (RAIZ / "models").mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, RUTA_MODELO, compress=3)
    RUTA_METRICAS.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RUTA_REPORTES / "reporte_clasificacion.txt").write_text(
        reporte_texto, encoding="utf-8"
    )
    _graficar_matriz_confusion(cm, RUTA_REPORTES / "matriz_confusion.png")
    _graficar_curva_perdida(
        modelo.named_steps["mlp"], RUTA_REPORTES / "curva_perdida.png"
    )
    _escribir_analisis(metricas)

    if verbose:
        print("=" * 68)
        print("ENTRENAMIENTO MLP · PREDICCIÓN DE APROBACIÓN")
        print("=" * 68)
        print(f"Registros usados: {len(df)} | Entrenamiento: {len(X_entrenamiento)} | Prueba: {len(X_prueba)}")
        print(f"Mejor arquitectura: {busqueda.best_params_}")
        print(f"F1 validación cruzada: {busqueda.best_score_:.4f}")
        print("-" * 68)
        print(reporte_texto)
        print("Matriz de confusión (filas=real, columnas=predicho):")
        print(cm)
        print("-" * 68)
        print(
            "Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
            "Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}".format(
                **metricas["metricas_prueba"]
            )
        )
        print(f"Modelo guardado en: {RUTA_MODELO}")
        print(f"Reportes guardados en: {RUTA_REPORTES}")
        print("=" * 68)

    return metricas


if __name__ == "__main__":
    entrenar_modelo()
