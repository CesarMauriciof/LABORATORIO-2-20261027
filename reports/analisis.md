# Análisis y conclusiones

Generado automáticamente el 2026-10-27 16:43 UTC por `scripts/entrenar_modelo.py`.

## Configuración del experimento

- Registros utilizados: **600** (300 aprobados / 300 reprobados).
- Entrenamiento: **480** registros · Prueba independiente: **120** registros (división estratificada 80/20, semilla 42).
- Arquitectura seleccionada: capas ocultas **32 → 16**, alpha **0.001**, tasa de aprendizaje **0.01**.
- Preprocesamiento dentro del pipeline: imputación por mediana → `StandardScaler` → MLP.
- `Calificacion_Final` se excluyó de las entradas porque es una salida del dataset y el objetivo `Aprobado` se deriva de ella (fuga de información).

## Métricas sobre el set de prueba

| Métrica | Valor |
| --- | --- |
| Accuracy | 0.9667 |
| Precision | 0.9828 |
| Recall | 0.9500 |
| F1-Score | 0.9661 |
| ROC-AUC | 0.9969 |

F1 promedio en validación cruzada (5 folds): **0.9857**.

## Conclusiones

1. **Rendimiento.** El MLP acierta el 96.7 % de los casos del set de prueba con un F1 de 0.966. La brecha entre validación cruzada (0.986) y prueba (0.966) es pequeña, por lo que no se observa sobreajuste relevante.
2. **Naturaleza del problema.** El objetivo se construyó a partir de una combinación lineal de las variables académicas, de modo que el problema es casi linealmente separable. Un MLP pequeño basta; añadir capas o neuronas no aporta mejoras y sí aumenta el riesgo de sobreajuste.
3. **Variables relevantes.** Tareas, Exámenes, Estudio_Horas y Asistencia concentran la señal. Sueno_Horas y Extracurriculares muestran una separación casi nula entre clases, por lo que el modelo no las necesita para decidir.
4. **Validez de las métricas.** Con clases balanceadas, accuracy y F1 son comparables; aun así se reportan precision y recall por clase para detectar errores asimétricos. La matriz de confusión confirma que los errores se reparten entre ambas clases.
5. **Limitaciones.** El dataset es sintético y el objetivo es determinista, por lo que un rendimiento cercano al 100 % no se trasladaría a estudiantes reales. Para un uso real harían falta datos más ricos, validación temporal y revisión de sesgos.

## Recomendaciones

- Usar el modelo como apoyo, nunca como decisión automática sobre un estudiante.
- Reentrenar y reevaluar si cambia la distribución de los datos o el criterio de aprobación.
- Documentar el pipeline completo (imputación y escalado incluidos) para evitar *data leakage* entre entrenamiento y producción.
