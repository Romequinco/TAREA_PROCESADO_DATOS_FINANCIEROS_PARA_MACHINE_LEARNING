# TAREA_PROCESADO_DATOS_FINANCIEROS_PARA_MACHINE_LEARNING
Entender distintas técnicas de preprocesado financiero. Para ello seguiremos las recomendaciones de MLdP en [1] y [2]
## Resumen del flujo completo del notebook

A continuación se resume, en forma de diagrama textual, el flujo de datos y decisiones a lo largo del notebook:

```text
0. Setup, parámetros y carga inicial de datos
   ├─ Definir entorno, rutas, fechas, semilla
   ├─ Cargar CSV desde Kaggle y filtrar ventana de 3 años
   └─ Explorar dataset crudo (integridad, tipos, NaNs)
      ↓
1. Construcción y comparación de barras alternativas
   ├─ Motivar el uso de barras no temporales
   ├─ Definir y construir tick bars, volume bars y dollar bars
   ├─ Comparar propiedades estadísticas y visuales
   └─ Seleccionar dollar bars como representación base
      ↓
2. Diferenciación fraccional de la serie de precios
   ├─ Introducir concepto y motivación de diferenciación fraccional
   ├─ Calcular series para varios valores de d (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
   ├─ Evaluar memoria vs estacionariedad (ACF, varianza, etc.)
   └─ Elegir valor de d y añadir la serie diferenciada al dataframe
      ↓
3. Construcción y limpieza de la matriz de covarianza
   ├─ Cargar precios wide desde `data/datos_crypto_limpios.parquet` (índice 5-min) con 10 cryptos (p=10)
   ├─ Construir X como retornos log por activo: `X = np.log(prices).diff().dropna(how="any")`
   ├─ Estandarizar (StandardScaler) y estimar covarianza empírica en los últimos 3 años por fecha
   ├─ Analizar espectro con Marchenko–Pastur (p=10, usando los parámetros efectivos del bulk; umbral MP `e_max`)
   ├─ Denoising con “Constant Residual” (López de Prado): ruido (λ ≤ e_max) → `noise_replacement = media_ruido` (promedio del bulk ruidoso)
   └─ Comparar covarianza original vs limpia
      ↓
4. Etiquetado de eventos mediante triple barrera
   ├─ Introducir triple barrera (take profit, stop loss, barrera temporal)
   ├─ Aplicar thresholds fijos (±1 %, ±2 %)
   ├─ Aplicar thresholds dinámicos basados en volatilidad rolling
   ├─ Generar y comparar distribuciones de etiquetas
   └─ Seleccionar esquema(s) de etiquetado para modelado
      ↓
5. Validación cruzada temporal y definición de splits
   ├─ Discutir limitaciones de la validación cruzada aleatoria
   ├─ Definir splits 70/30, 80/20, 90/10 respetando el tiempo
   ├─ Visualizar train/test sobre la serie de barras etiquetada
   └─ Dejar definidos los índices de train/test para futuros modelos
```

Este diagrama sirve como referencia rápida para entender cómo cada sección transforma los datos y qué decisiones alimentan las etapas posteriores del pipeline.

