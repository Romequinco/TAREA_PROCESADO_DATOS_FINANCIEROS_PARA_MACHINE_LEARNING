# Guion de video (<= 5 minutos): Preprocesado de datos financieros

## 0) Resumen ejecutivo (20-30 segundos)
- Objetivo: transformar BTC/USDT de 1 minuto en un dataset robusto para ML financiero, respetando la estructura temporal.

## 1) Fases del preprocesado (nucleo del video)

### Fase 1: Carga y limpieza base de datos
**Que se hace (en 1-2 frases):**  
Se descarga desde kaggle el dataset de BTC,y se normaliza a indice `datetime`. Se estandarizan columnas OHLCV, se filtra el rango de 3 años y se valida calidad.

**Por que se usa esta tecnica aqui:**  
Asegura consistencia del dato antes del preprocesado cuantitativo.

**Tiempo sugerido de narracion:** 40 segundos

---

### Fase 2: Barras alternativas y seleccion base
**Que se hace (en 1-2 frases):**  
Se construyen tick, volume y dollar bars, y se comparan con metricas y graficos. La fase concluye seleccionando dollar bars.

**Tecnica de Marcos Lopez de Prado usada:**  
Event-based bars (tick/volume/dollar).

**Pasos internos de la fase (max. 4 bullets):**
- Parametros: `TICK_SIZE=10`, `VOLUME_THRESHOLD=10`, `DOLLAR_THRESHOLD=750000`.
- Construccion de `df_tick`, `df_volume`, `df_dollar`.
- Comparacion de numero de barras y distribucion de retornos.
- Decision final: `df_bars = df_dollar`.

**Por que se usa esta tecnica aqui:**  
Muestrea por actividad de mercado y no por tiempo fijo.

**Ventajas (max. 3 bullets):**
- Reduce sesgo de barras temporales.
- Dollar bars capturan mejor actividad economica.
- Mejor base para etiquetado posterior.

**Limitaciones / riesgos (max. 3 bullets):**
- Sensible a umbrales elegidos.
- No hay ticks reales, se parte de 1 minuto.
- Evaluacion principalmente descriptiva.

**Tiempo sugerido de narracion:** 50 segundos

---

### Fase 3: Diferenciacion fraccional (FFD)
**Que se hace (en 1-2 frases):**  
Se implementa FFD, se prueban varios `d` y se evaluan con ADF y correlacion con la serie original. Se selecciona `d=0.4`.

**Tecnica de Marcos Lopez de Prado usada:**  
Fractional Differentiation (FFD).

**Pasos internos de la fase (max. 4 bullets):**
- Barrido de `d` en `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- Calculo de p-value ADF y correlacion memoria.
- Criterio: menor `d` estacionario.
- Resultado: `D_OPTIMO = D_ALTERNATIVO = 0.4`; salida `close_fd`.

**Por que se usa esta tecnica aqui:**  
Busca estacionariedad sin destruir informacion temporal util.

**Ventajas (max. 3 bullets):**
- Trade-off explicito memoria/estacionariedad.
- Seleccion basada en evidencia estadistica.
- Evita sobrediferenciacion.

**Limitaciones / riesgos (max. 3 bullets):**
- Depende de muestra y significancia.
- Implementacion manual puede ser lenta.
- El `d` optimo puede variar fuera de muestra.

**Tiempo sugerido de narracion:** 50 segundos

---

### Fase 4: Limpieza de covarianza multiactivo
**Que se hace (en 1-2 frases):**  
Con un parquet de 10 criptos a 5 minutos, se limpia correlacion/covarianza con Marchenko-Pastur y clipping de autovalores.

**Tecnica de Marcos Lopez de Prado usada:**  
Marchenko-Pastur y clipping

**Pasos internos de la fase (max. 4 bullets):**
- Retornos logaritmicos (`T=581488`, `N=10`).
- Matrices ruidosas de correlacion/covarianza.
- Denoising espectral y reconstruccion de matriz limpia.
- Mejora del numero de condicion: 81.61%.

**Por que se usa esta tecnica aqui:**  
Disminuye ruido y estabiliza matrices para modelado financiero.

**Ventajas (max. 3 bullets):**
- Menos correlaciones espurias.
- Mayor estabilidad numerica.
- Mejor base para enfoques multiactivo.

**Limitaciones / riesgos (max. 3 bullets):**
- Supuestos MP pueden no cumplirse siempre.
- Sensible a la relacion `T/N`.
- Rama separada del flujo principal de etiquetas BTC.

**Tiempo sugerido de narracion:** 40 segundos

---

### Fase 5: Etiquetado con triple barrera
**Que se hace (en 1-2 frases):**  
Se etiqueta cada evento por la primera barrera tocada (+1, 0, -1), comparando umbral fijo 1%, fijo 2% y dinamico por volatilidad.

**Tecnica de Marcos Lopez de Prado usada:**  
Triple Barrier Method

**Pasos internos de la fase (max. 4 bullets):**
- Parametros: `BARRIER_WINDOW=20`, `THRESHOLD_1=1%`, `THRESHOLD_2=2%`, `VOL_MULTIPLIER=1.5`.
- Generacion de `df_labels` con tres esquemas.
- Comparacion de distribuciones y contingencias.
- Decision final: `labels_main = labels_1pct`.

**Por que se usa esta tecnica aqui:**  
Captura trayectoria real del precio y horizonte temporal.

**Ventajas (max. 3 bullets):**
- Mejor que retorno fijo de horizonte unico.
- Respeta causalidad temporal.
- Permite sensibilidad por umbral.

**Limitaciones / riesgos (max. 3 bullets):**
- Desbalance de clases (predomina clase 0 en 1%).
- Sensible a ventana y umbrales.
- Umbral dinamico depende de volatilidad estimada.

**Tiempo sugerido de narracion:** 50 segundos

---

### Fase 6: Validacion temporal con purga y embargo
**Que se hace (en 1-2 frases):**  
Se definen y visualizan splits temporales con purga y embargo para reducir leakage entre train y test.

**Tecnica de Marcos Lopez de Prado usada:**  
Purged K-Fold / Embargo

**Pasos internos de la fase (max. 4 bullets):**
- Construccion de `t1` aproximado como `t0 + BARRIER_WINDOW`.
- Funciones para train/test/purga/embargo.
- Visualizacion de configuraciones 70/30, 80/20, 90/10 y proporciones fijas.
- Enfoque de diseno de validacion, no de entrenamiento.

**Por que se usa esta tecnica aqui:**  
Evita contaminacion temporal al evaluar futuros modelos.

**Ventajas (max. 3 bullets):**
- Protocolo mas realista en series financieras.
- Reduce dependencia entre folds.
- Facilita trazabilidad del split.

**Limitaciones / riesgos (max. 3 bullets):**
- No es implementacion formal completa de `PurgedKFold` de AFML.
- `t1` se aproxima por timeout.
- No se reportan metricas de modelo.

**Tiempo sugerido de narracion:** 40 segundos

---
