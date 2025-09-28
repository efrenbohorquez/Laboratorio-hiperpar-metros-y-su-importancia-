# Optimización de Hiperparámetros con Keras Tuner

Este proyecto implementa un análisis completo de optimización de hiperparámetros para un modelo de clasificación binaria utilizando el dataset Breast Cancer Wisconsin. Se comparan dos algoritmos avanzados de optimización: Hyperband y Optimización Bayesiana.

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
- Teoría del algoritmo Hyperband
- Configuración y ejecución de la búsqueda
- Análisis de resultados y visualizaciones

### Ejercicio 2: Implementación de Optimización Bayesiana
- Teoría de la optimización bayesiana
- Componentes fundamentales (Gaussian Process, Función de Adquisición)
- Configuración y ejecución de la búsqueda
- Análisis comparativo con Hyperband

### Ejercicio 3: Visualización Avanzada de Resultados
- Importancia de la visualización en optimización
- Gráficos comparativos de rendimiento
- Análisis de eficiencia temporal
- Interpretación de resultados

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

## 🎯 Uso

1. Abrir el notebook `optimizacion_hiperparametros_keras_tuner.ipynb`
2. Ejecutar las celdas en orden
3. Los resultados de la optimización se guardan automáticamente en directorios separados

## 📊 Resultados Esperados

- Comparación de rendimiento entre Hyperband y Optimización Bayesiana
- Mejores hiperparámetros encontrados para el dataset
- Visualizaciones detalladas del proceso de optimización
- Análisis de eficiencia computacional

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

## 🤝 Contribuciones

Este proyecto sigue las mejores prácticas modernas de machine learning y está diseñado para fines educativos y de investigación.

## 📄 Licencia

Este proyecto es de uso educativo.