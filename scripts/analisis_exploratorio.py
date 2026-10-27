"""Análisis exploratorio del dataset de estudiantes.

Genera estadísticas descriptivas, verifica el balance de clases y crea las
figuras que sustentan las conclusiones del informe.

Uso:
    python scripts/analisis_exploratorio.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "dataset_estudiantes_balanceado.csv"
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
ETIQUETAS = {0: "Reprobado", 1: "Aprobado"}
COLORES = {0: "#d1495b", 1: "#2a9d8f"}


def _graficar_balance(df: pd.DataFrame, ruta: Path) -> None:
    conteos = df[OBJETIVO].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    barras = ax.bar(
        [ETIQUETAS[int(i)] for i in conteos.index],
        conteos.values,
        color=[COLORES[int(i)] for i in conteos.index],
        width=0.55,
    )
    ax.bar_label(barras, fmt="%d", padding=4, fontsize=11)
    ax.set_title("Balance de clases")
    ax.set_ylabel("Estudiantes")
    ax.set_ylim(0, conteos.max() * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _graficar_distribuciones(df: pd.DataFrame, ruta: Path) -> None:
    fig, ejes = plt.subplots(2, 3, figsize=(13, 7))
    for eje, columna in zip(ejes.ravel(), CARACTERISTICAS):
        for clase in (0, 1):
            datos = df.loc[df[OBJETIVO] == clase, columna].dropna()
            eje.hist(
                datos,
                bins=20,
                alpha=0.6,
                color=COLORES[clase],
                label=ETIQUETAS[clase],
                density=True,
            )
        eje.set_title(columna)
        eje.set_ylabel("Densidad")
        eje.grid(alpha=0.2)
    ejes[0, 0].legend(frameon=False)
    fig.suptitle("Distribución de cada variable según la clase", fontsize=14)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _graficar_correlacion(df: pd.DataFrame, ruta: Path) -> None:
    columnas = CARACTERISTICAS + ["Calificacion_Final", OBJETIVO]
    correlacion = df[columnas].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    imagen = ax.imshow(correlacion, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columnas)), columnas, rotation=45, ha="right")
    ax.set_yticks(range(len(columnas)), columnas)
    for i in range(len(columnas)):
        for j in range(len(columnas)):
            ax.text(
                j,
                i,
                f"{correlacion.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(correlacion.iloc[i, j]) > 0.55 else "#222222",
            )
    ax.set_title("Correlación de Pearson")
    fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not RUTA_DATOS.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {RUTA_DATOS}")

    RUTA_REPORTES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUTA_DATOS)

    descriptivas = df[CARACTERISTICAS + ["Calificacion_Final"]].describe().T
    descriptivas["mediana"] = df[CARACTERISTICAS + ["Calificacion_Final"]].median()
    descriptivas = descriptivas[
        ["count", "mean", "std", "min", "25%", "mediana", "75%", "max"]
    ].round(3)
    descriptivas.index.name = "variable"
    descriptivas.to_csv(RUTA_REPORTES / "estadisticas_descriptivas.csv")

    nulos = df.isna().sum()
    conteos = df[OBJETIVO].value_counts().sort_index()
    correlacion_objetivo = (
        df[CARACTERISTICAS + ["Calificacion_Final", OBJETIVO]]
        .corr(numeric_only=True)[OBJETIVO]
        .drop(OBJETIVO)
        .sort_values(key=np.abs, ascending=False)
    )

    _graficar_balance(df, RUTA_REPORTES / "eda_balance_clases.png")
    _graficar_distribuciones(df, RUTA_REPORTES / "eda_distribuciones.png")
    _graficar_correlacion(df, RUTA_REPORTES / "eda_correlacion.png")

    lineas = [
        "# Análisis exploratorio",
        "",
        f"- Registros: **{len(df)}**",
        f"- Columnas: **{len(df.columns)}**",
        f"- Valores ausentes totales: **{int(nulos.sum())}**",
        f"- Balance: **{int(conteos.get(1, 0))} aprobados** y **{int(conteos.get(0, 0))} reprobados** "
        f"({conteos.get(1, 0) / len(df):.1%} / {conteos.get(0, 0) / len(df):.1%}).",
        "",
        "## Estadísticas descriptivas",
        "",
        "```",
        descriptivas.to_string(),
        "```",
        "",
        "## Correlación con la variable objetivo",
        "",
        "| Variable | Correlación con Aprobado |",
        "| --- | --- |",
    ]
    for variable, valor in correlacion_objetivo.items():
        lineas.append(f"| {variable} | {valor:+.3f} |")

    lineas += [
        "",
        "## Hallazgos",
        "",
        "1. El dataset ya está balanceado 50/50, por lo que el submuestreo no es necesario "
        "(el script de entrenamiento lo verifica y solo lo aplicaría si hiciera falta).",
        "2. No hay valores ausentes; aun así, el pipeline conserva la imputación por mediana "
        "para que el modelo funcione con datos incompletos en producción.",
        "3. `Calificacion_Final` correlaciona fuertemente con `Aprobado` (r ≈ 0,80), porque el "
        "objetivo se deriva de ella. Incluirla como entrada sería fuga de información, así que se excluye.",
        "4. Las variables académicas (Tareas, Exámenes, Asistencia) y las horas de estudio son las "
        "que más se separan entre clases.",
        "5. `Sueno_Horas` y `Extracurriculares` muestran correlaciones cercanas a cero y distribuciones "
        "superpuestas: aportan poca señal al problema.",
        "",
        "## Figuras",
        "",
        "- `eda_balance_clases.png`",
        "- `eda_distribuciones.png`",
        "- `eda_correlacion.png`",
        "",
    ]
    (RUTA_REPORTES / "eda_resumen.md").write_text("\n".join(lineas), encoding="utf-8")

    print(descriptivas)
    print()
    print("Balance de clases:")
    print(conteos.rename(index=ETIQUETAS))
    print()
    print("Correlación con Aprobado:")
    print(correlacion_objetivo.round(3))
    print()
    print(f"Reportes guardados en: {RUTA_REPORTES}")


if __name__ == "__main__":
    main()
