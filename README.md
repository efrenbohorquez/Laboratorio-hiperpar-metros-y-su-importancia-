# Optimización de Hiperparámetros con Keras Tuner

## 🎓 ¿Qué es este Laboratorio?

Este laboratorio es una guía práctica y completa que **explica** y demuestra la importancia de la optimización de hiperparámetros en modelos de Deep Learning. A través de ejemplos prácticos y ejercicios interactivos, los estudiantes aprenderán cómo ajustar automáticamente los hiperparámetros de redes neuronales para mejorar significativamente el rendimiento de sus modelos.

El proyecto implementa un análisis completo de optimización de hiperparámetros para un modelo de clasificación binaria utilizando el dataset Breast Cancer Wisconsin. Se comparan dos algoritmos avanzados de optimización: **Hyperband** y **Optimización Bayesiana**.

## 🎯 Objetivos de Aprendizaje

Al completar este laboratorio, los estudiantes serán capaces de:

1. **Comprender** la diferencia entre parámetros e hiperparámetros en modelos de Deep Learning
2. **Explicar** por qué la optimización de hiperparámetros es crucial para el rendimiento del modelo
3. **Implementar** búsquedas automáticas de hiperparámetros usando Keras Tuner
4. **Comparar** diferentes algoritmos de optimización (Hyperband vs Bayesian Optimization)
5. **Analizar** y visualizar resultados de experimentos de optimización
6. **Aplicar** estas técnicas a problemas reales de clasificación

## 📚 ¿Qué Explica este Laboratorio?

Este laboratorio explica de manera detallada y práctica los siguientes conceptos fundamentales:

### 1. Conceptos Teóricos Fundamentales

- **¿Qué son los hiperparámetros?** - Explicación clara de cómo se diferencian de los parámetros del modelo
- **¿Por qué son importantes?** - Demostración de cómo pueden mejorar el rendimiento en 5-15% o más (mejora relativa en accuracy comparada con hiperparámetros por defecto)
- **Métodos de optimización** - Comparación entre técnicas manuales, Grid Search, Random Search y métodos modernos

### 2. Algoritmos de Optimización Modernos

#### Hyperband
- **Explicación del algoritmo:** Cómo funciona la asignación adaptativa de recursos
- **Ventajas:** Por qué es eficiente computacionalmente
- **Cuándo usar:** Ideal para exploración inicial rápida

#### Optimización Bayesiana
- **Explicación del algoritmo:** Cómo modela la función objetivo usando Gaussian Processes
- **Ventajas:** Por qué encuentra mejores configuraciones
- **Cuándo usar:** Ideal para refinamiento fino y obtener el mejor rendimiento

### 3. Caso de Estudio Práctico: Clasificación de Cáncer de Mama

El laboratorio utiliza un problema real de clasificación médica para:
- Demostrar el impacto real de la optimización
- Mostrar cómo preparar y analizar datos
- Explicar el proceso completo de desarrollo de un modelo

## 📋 Contenido del Notebook

### Introducción Teórica
- Conceptos fundamentales de hiperparámetros
- Diferencias entre parámetros e hiperparámetros
- Importancia de la optimización de hiperparámetros
- Comparación entre métodos tradicionales y Keras Tuner

### Preparación y Análisis del Dataset
- Carga y exploración del Breast Cancer Wisconsin dataset
- Análisis exploratorio de datos (EDA)
- Visualizaciones del dataset
- Preprocesamiento y estandarización

### Función de Construcción del Modelo
- Arquitectura variable de red neuronal
- Hiperparámetros optimizables:
  - Número de capas ocultas (1-5)
  - Unidades por capa (32-512)
  - Funciones de activación (ReLU, Tanh, Sigmoid)
  - Regularización L2
  - Tasa de dropout
  - Optimizador (Adam, SGD, RMSprop)

### Ejercicio 1: Implementación de Hyperband
- **Teoría explicada:** Cómo Hyperband asigna recursos adaptativamente
- Configuración y ejecución de la búsqueda
- Análisis de resultados y visualizaciones
- **Interpretación:** Qué significan los resultados y cómo usarlos

### Ejercicio 2: Implementación de Optimización Bayesiana
- **Teoría explicada:** Fundamentos de la optimización bayesiana
- Componentes fundamentales (Gaussian Process, Función de Adquisición)
- Configuración y ejecución de la búsqueda
- **Comparación explicada:** Por qué obtiene diferentes resultados que Hyperband

### Ejercicio 3: Visualización Avanzada de Resultados
- Importancia de la visualización en optimización
- Gráficos comparativos de rendimiento
- Análisis de eficiencia temporal
- **Interpretación práctica:** Cómo tomar decisiones basadas en los resultados

## 🚀 Requisitos del Sistema

- Python 3.7+
- TensorFlow 2.x
- Keras Tuner
- Scikit-learn
- NumPy, Pandas, Matplotlib, Seaborn

## 📦 Instalación

```bash
pip install -r requirements.txt
```

## 🎯 Cómo Usar este Laboratorio

### Opción 1: Jupyter Notebook (Recomendado para Aprendizaje)

1. **Abrir el notebook:** `optimizacion_hiperparametros_keras_tuner.ipynb`
2. **Leer cada sección cuidadosamente:** El notebook explica cada concepto antes de implementarlo
3. **Ejecutar las celdas en orden:** Cada celda construye sobre la anterior
4. **Experimentar:** Modificar parámetros y ver cómo cambian los resultados
5. **Revisar visualizaciones:** Cada gráfico explica un aspecto diferente del proceso

Los resultados de la optimización se guardan automáticamente en directorios separados (`hyperband_tuning/` y `bayesian_tuning/`).

### Opción 2: Script Python (Para Ejecución Automática)

```bash
python optimizacion_hiperparametros_keras_tuner.py
```

### Guía Paso a Paso

#### Paso 1: Introducción Teórica (15 minutos)
- Lee la sección de conceptos fundamentales
- Comprende la diferencia entre parámetros e hiperparámetros
- Revisa la tabla comparativa de métodos de optimización

#### Paso 2: Análisis del Dataset (10 minutos)
- Explora las características del Breast Cancer dataset
- Observa las visualizaciones de distribución de clases
- Comprende la importancia del preprocesamiento

#### Paso 3: Construcción del Modelo (15 minutos)
- Estudia la función `build_model(hp)`
- Identifica qué hiperparámetros se están optimizando
- Entiende cómo se define el espacio de búsqueda

#### Paso 4: Ejercicio Hyperband (20-30 minutos)
- Lee la explicación del algoritmo
- Ejecuta la búsqueda de hiperparámetros
- Analiza los resultados y visualizaciones
- **Tiempo de ejecución estimado:** ~15 minutos

#### Paso 5: Ejercicio Bayesian Optimization (20-30 minutos)
- Lee la explicación del algoritmo bayesiano
- Ejecuta la búsqueda de hiperparámetros
- Compara resultados con Hyperband
- **Tiempo de ejecución estimado:** ~20 minutos

#### Paso 6: Análisis Comparativo (10 minutos)
- Revisa las visualizaciones comparativas
- Interpreta las métricas de rendimiento
- Comprende las ventajas y desventajas de cada método

**Tiempo total estimado:** 2-2.5 horas (incluyendo ejecución de experimentos)

## 📊 Resultados Esperados y su Interpretación

### Lo que Aprenderás a Interpretar:

#### 1. Comparación de Rendimiento
- **Hyperband:** Típicamente alcanza ~96% de precisión en ~15 minutos (valores aproximados, varían según hardware)
- **Bayesian Optimization:** Típicamente alcanza ~97% de precisión en ~20 minutos (valores aproximados, varían según hardware)
- **Interpretación:** Bayesian Optimization es más preciso pero requiere más tiempo
- **Nota:** Los tiempos de ejecución son aproximados y dependen del hardware (CPU vs GPU, velocidad de procesador)

#### 2. Mejores Hiperparámetros Encontrados
Aprenderás a identificar:
- La arquitectura óptima de red (número de capas y neuronas)
- Las mejores funciones de activación para este problema
- Los valores óptimos de regularización (dropout, L2)
- El mejor optimizador y learning rate

#### 3. Visualizaciones que Explican el Proceso
El laboratorio genera gráficos que explican:
- **Evolución del rendimiento:** Cómo mejora el modelo con cada trial
- **Comparación temporal:** Eficiencia de cada algoritmo
- **Distribución de hiperparámetros:** Qué valores funcionan mejor
- **Trade-offs:** Relación entre precisión y tiempo de cómputo

#### 4. Análisis de Eficiencia Computacional
Comprenderás:
- Cómo medir la eficiencia (score por minuto)
- Cuándo usar cada algoritmo según tus recursos
- Cómo balancear precisión vs tiempo de ejecución

## 🔧 Configuración de la Optimización

### Hyperband
- Máximo 50 épocas
- Factor de reducción: 3
- 2 iteraciones de Hyperband

### Optimización Bayesiana
- Máximo 25 trials
- 5 puntos iniciales de exploración
- Parámetros GP: alpha=1e-4, beta=2.6

## 📈 Métricas de Evaluación

- Accuracy de validación
- Precision y Recall
- Tiempo de ejecución
- Eficiencia (score por minuto)

## 💡 Conceptos Clave Explicados

### 1. Hiperparámetros vs Parámetros

Este laboratorio explica claramente que:
- **Parámetros** (pesos, sesgos) se aprenden durante el entrenamiento
- **Hiperparámetros** (arquitectura, learning rate) se definen antes del entrenamiento
- La optimización de hiperparámetros es un **meta-aprendizaje** sobre el proceso de aprendizaje

### 2. ¿Por Qué es Importante la Optimización?

El laboratorio demuestra que:
- Un modelo con hiperparámetros mal configurados puede tener 70-80% de precisión
- El mismo modelo con hiperparámetros optimizados puede alcanzar 95-97% de precisión
- **Diferencia:** 15-27% de mejora en rendimiento simplemente optimizando configuraciones

### 3. Métodos de Optimización Explicados

#### Manual Tuning
- **Qué es:** Ajuste basado en experiencia
- **Problema:** Lento, subjetivo, limitado
- **Cuándo usar:** Prototipos rápidos o problemas muy simples

#### Grid Search
- **Qué es:** Probar todas las combinaciones en una grilla
- **Problema:** Crece exponencialmente (ejemplo: 10 parámetros con 3 valores cada uno = 3^10 = 59,049 combinaciones)
- **Cuándo usar:** Pocos hiperparámetros y recursos abundantes

#### Random Search
- **Qué es:** Probar combinaciones aleatorias
- **Ventaja:** Más eficiente que Grid Search en alta dimensionalidad
- **Cuándo usar:** Primera exploración con muchos hiperparámetros

#### Hyperband (Moderno)
- **Qué es:** Asignación adaptativa de recursos
- **Ventaja:** Elimina configuraciones malas tempranamente
- **Cuándo usar:** Búsqueda inicial eficiente con presupuesto limitado

#### Bayesian Optimization (Moderno)
- **Qué es:** Modelo probabilístico de la función objetivo
- **Ventaja:** Aprende de evaluaciones anteriores
- **Cuándo usar:** Refinamiento fino para obtener el mejor rendimiento

## 🎓 Aplicaciones Prácticas

Después de completar este laboratorio, podrás:

1. **Mejorar modelos existentes:** Aplicar estas técnicas a tus propios proyectos
2. **Ahorrar tiempo:** Automatizar la búsqueda en lugar de ajustar manualmente
3. **Justificar decisiones:** Explicar por qué elegiste ciertos hiperparámetros
4. **Optimizar recursos:** Balancear precisión vs costo computacional
5. **Investigar eficientemente:** Probar muchas configuraciones sistemáticamente

## 🤝 Contribuciones

Este proyecto sigue las mejores prácticas modernas de machine learning y está diseñado para fines educativos y de investigación.

## ❓ Preguntas Frecuentes

### ¿Necesito experiencia previa en Deep Learning?
**Sí, nivel básico.** El laboratorio asume que conoces:
- Conceptos básicos de redes neuronales
- Python y bibliotecas como NumPy/Pandas
- Fundamentos de machine learning (train/test split, métricas)

### ¿Cuánto tiempo toma completar el laboratorio?
**2-2.5 horas** incluyendo:
- Lectura de conceptos: 40 minutos
- Ejecución de experimentos: 35-50 minutos (depende del hardware)
- Análisis de resultados: 45 minutos

### ¿Necesito GPU para ejecutarlo?
**No es necesario.** El dataset es pequeño (569 muestras) y los modelos entrenan rápido en CPU. Sin embargo, con GPU será más rápido.

### ¿Puedo modificar el código para mi propio dataset?
**¡Sí!** El código está diseñado para ser adaptable. Necesitarás:
1. Cargar tu dataset en lugar del Breast Cancer
2. Ajustar la arquitectura en `build_model()` si es necesario
3. Modificar los rangos de hiperparámetros según tu problema

### ¿Qué hacer si los experimentos fallan?
Verifica:
1. Todas las dependencias están instaladas: `pip install -r requirements.txt`
2. Tienes suficiente espacio en disco (los resultados se guardan en carpetas)
3. La versión de TensorFlow es compatible (2.x)

### ¿Cómo interpreto los resultados?
El laboratorio incluye secciones detalladas que explican:
- Qué significan las métricas
- Cómo comparar algoritmos
- Cuándo elegir un método sobre otro
- Cómo usar los mejores hiperparámetros encontrados

## 📄 Licencia

Este proyecto es de uso educativo.