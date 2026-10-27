# Análisis exploratorio

- Registros: **600**
- Columnas: **8**
- Valores ausentes totales: **0**
- Balance: **300 aprobados** y **300 reprobados** (50.0% / 50.0%).

## Estadísticas descriptivas

```
                    count    mean     std     min     25%  mediana     75%      max
variable                                                                           
Tareas              600.0  65.034  20.436  30.354  47.423   65.988  82.684   99.980
Examenes            600.0  59.623  23.277  20.371  38.622   60.025  79.822   99.835
Asistencia          600.0  69.613  17.139  40.296  54.253   69.806  83.104   99.965
Estudio_Horas       600.0  13.515   6.499   2.036   7.986   13.791  18.931   24.962
Sueno_Horas         600.0   6.524   1.468   4.000   5.348    6.491   7.828    8.989
Extracurriculares   600.0   0.407   0.492   0.000   0.000    0.000   1.000    1.000
Calificacion_Final  600.0  74.669  14.506  37.548  64.357   74.674  84.145  110.833
```

## Correlación con la variable objetivo

| Variable | Correlación con Aprobado |
| --- | --- |
| Calificacion_Final | +0.795 |
| Estudio_Horas | +0.520 |
| Examenes | +0.462 |
| Tareas | +0.326 |
| Asistencia | +0.130 |
| Extracurriculares | -0.048 |
| Sueno_Horas | -0.012 |

## Hallazgos

1. El dataset ya está balanceado 50/50, por lo que el submuestreo no es necesario (el script de entrenamiento lo verifica y solo lo aplicaría si hiciera falta).
2. No hay valores ausentes; aun así, el pipeline conserva la imputación por mediana para que el modelo funcione con datos incompletos en producción.
3. `Calificacion_Final` correlaciona fuertemente con `Aprobado` (r ≈ 0,80), porque el objetivo se deriva de ella. Incluirla como entrada sería fuga de información, así que se excluye.
4. Las variables académicas (Tareas, Exámenes, Asistencia) y las horas de estudio son las que más se separan entre clases.
5. `Sueno_Horas` y `Extracurriculares` muestran correlaciones cercanas a cero y distribuciones superpuestas: aportan poca señal al problema.

## Figuras

- `eda_balance_clases.png`
- `eda_distribuciones.png`
- `eda_correlacion.png`
