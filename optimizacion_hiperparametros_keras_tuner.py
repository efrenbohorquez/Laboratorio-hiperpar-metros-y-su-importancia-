#!/usr/bin/env python
# coding: utf-8

# 📚 Introducción Teórica
# ¿Qué son los Hiperparámetros?
# Los hiperparámetros son configuraciones que definen la arquitectura y el comportamiento de un modelo de aprendizaje automático, pero que no se aprenden durante el entrenamiento. A diferencia de los parámetros (como pesos y sesgos), los hiperparámetros deben ser establecidos antes del entrenamiento.
# 
# 🔍 Diferencias Clave: Parámetros vs Hiperparámetros
# Aspecto	Parámetros	Hiperparámetros
# Definición	Variables aprendidas por el modelo	Configuraciones establecidas antes del entrenamiento
# Ejemplos	Pesos, sesgos	Learning rate, número de capas, dropout rate
# Optimización	Gradient descent, backpropagation	Grid search, random search, Bayesian optimization
# Modificación	Durante el entrenamiento	Antes del entrenamiento
# 
# ⚠️ Importancia de la Optimización de Hiperparámetros
# Rendimiento: Puede mejorar la precisión del modelo en 5-15%
# Generalización: Reduce overfitting y mejora la capacidad de generalización
# Eficiencia: Optimiza el tiempo de entrenamiento y los recursos computacionales
# Robustez: Hace el modelo más estable ante variaciones en los datos
# 
# 🛠️ Métodos Tradicionales vs Keras Tuner
# Métodos Tradicionales:
# 
# Manual: Ajuste basado en experiencia e intuición
# Grid Search: Búsqueda exhaustiva en una grilla predefinida
# Random Search: Selección aleatoria de combinaciones
# 
# Ventajas de Keras Tuner:
# 
# 🔧 Facilidad de uso: API simple y consistente
# 🚀 Algoritmos avanzados: Hyperband, Bayesian Optimization
# 📊 Integración nativa: Funciona perfectamente con Keras/TensorFlow
# 💾 Persistencia automática: Guarda resultados y permite reanudar búsquedas
# 📈 Visualización: Herramientas integradas para análisis de resultados

# In[ ]:


# 📦 Instalación de Keras Tuner y dependencias
# get_ipython().system('pip install -q keras_tuner')
# get_ipython().system('pip install -q seaborn')
print("✅ Instalación completada exitosamente!")


# In[ ]:


# 📚 Importación de librerías esenciales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt

# Librerías de sklearn para datos y preprocesamiento
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Configuración para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Configuración de matplotlib para mejores gráficos
plt.style.use('default')
sns.set_palette("husl")

print("🎯 Librerías importadas correctamente")
print(f"📊 TensorFlow version: {tf.__version__}")
print(f"🔧 Keras Tuner version: {kt.__version__}")


# 📊 Preparación y Análisis del Dataset
# Utilizaremos el Breast Cancer Wisconsin Dataset, un dataset clásico para clasificación binaria que contiene características extraídas de imágenes digitalizadas de masas de tejido mamario.
# 
# 🔬 Características del Dataset
# Instancias: 569 muestras
# Features: 30 características numéricas
# Clases: Maligno (1) y Benigno (0)
# Tipo: Problema de clasificación binaria

# In[ ]:


# 📥 Carga y exploración del dataset
data = load_breast_cancer()
X = data.data
y = data.target

print("🔍 ANÁLISIS EXPLORATORIO DEL DATASET")
print("=" * 50)
print(f"📈 Forma del dataset: {X.shape}")
print(f"🎯 Clases: {data.target_names}")
print(f"📊 Distribución de clases: {np.bincount(y)}")
print(f"📋 Características: {len(data.feature_names)}")

# Crear DataFrame para mejor visualización
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

print("\n📋 ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe().round(2))

print(f"\n🎯 BALANCE DE CLASES:")
print(f"Benigno (0): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
print(f"Maligno (1): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")


# In[ ]:


# 📊 Visualización del dataset
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribución de clases
axes[0, 0].pie([212, 357], labels=['Maligno', 'Benigno'], autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'])
axes[0, 0].set_title('🎯 Distribución de Clases')

# Histograma de algunas características importantes
axes[0, 1].hist([df[df['target']==0]['mean radius'], df[df['target']==1]['mean radius']], alpha=0.7, label=['Maligno', 'Benigno'], bins=20)
axes[0, 1].set_title('📏 Distribución del Radio Medio')
axes[0, 1].set_xlabel('Radio Medio')
axes[0, 1].legend()

# Correlación entre algunas características
correlation_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
corr_matrix = df[correlation_features + ['target']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
axes[1, 0].set_title('🔥 Mapa de Correlación')

# Boxplot de características importantes
df_melted = df[['mean radius', 'mean texture', 'target']].melt(id_vars=['target'])
sns.boxplot(data=df_melted, x='variable', y='value', hue='target', ax=axes[1, 1])
axes[1, 1].set_title('📦 Distribución por Clase')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("✅ Visualización del dataset completada")


# In[ ]:


# 🔧 División y preprocesamiento de datos
print("🔄 PREPROCESAMIENTO DE DATOS")
print("=" * 40)

# División estratificada del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"📊 Conjunto de entrenamiento: {X_train.shape}")
print(f"🧪 Conjunto de prueba: {X_test.shape}")

# Verificar distribución en conjuntos
print(f"\n🎯 Distribución en entrenamiento:")
print(f" Benigno: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
print(f" Maligno: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")

print(f"\n🎯 Distribución en prueba:")
print(f" Benigno: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
print(f" Maligno: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")

# Estandarización de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📐 Estadísticas después de la estandarización:")
print(f" Media del conjunto de entrenamiento: {X_train_scaled.mean():.3f}")
print(f" Desviación estándar del entrenamiento: {X_train_scaled.std():.3f}")

# Visualizar el efecto de la estandarización
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(X_train[:, 0], bins=30, alpha=0.7, label='Original')
ax1.set_title('📊 Antes de Estandarización')
ax1.set_xlabel('Valores')
ax1.set_ylabel('Frecuencia')
ax2.hist(X_train_scaled[:, 0], bins=30, alpha=0.7, label='Estandarizado', color='orange')
ax2.set_title('📊 Después de Estandarización')
ax2.set_xlabel('Valores')
ax2.set_ylabel('Frecuencia')
plt.tight_layout()
plt.show()

print("✅ Preprocesamiento completado exitosamente")


# 🏗️ Función de Construcción del Modelo Base
# Definiremos una función que construye modelos con arquitectura variable, permitiendo ajustar múltiples hiperparámetros simultáneamente.
# 
# 💡 Hiperparámetros a Optimizar
# Arquitectura: Número de capas ocultas (1-5)
# Neuronas: Unidades por capa (32-512)
# Activación: Funciones (ReLU, Tanh, Sigmoid)
# Regularización: L2 regularization (1e-5 to 1e-2)
# Dropout: Tasa de dropout (0.0-0.5)
# Optimizador: Adam, SGD, RMSprop

# In[ ]:


def build_model(hp):
    """ 🏗️ Construye un modelo de red neuronal con hiperparámetros variables

    Args:
        hp: Objeto HyperParameters de Keras Tuner

    Returns:
        model: Modelo compilado de Keras
    """

    # 🧱 Inicializar modelo secuencial
    model = keras.Sequential()

    # 📏 Definir número de capas ocultas
    num_layers = hp.Int(
        name='num_layers',
        min_value=1,
        max_value=5,
        default=2
    )

    # 🎯 Primera capa (incluye input_shape)
    model.add(layers.Dense(
        units=hp.Int(
            name='units_0',
            min_value=32,
            max_value=512,
            step=32,
            default=128
        ),
        activation=hp.Choice(
            name='activation_0',
            values=['relu', 'tanh', 'sigmoid'],
            default='relu'
        ),
        kernel_regularizer=keras.regularizers.l2(
            hp.Float(
                name='l2_0',
                min_value=1e-5,
                max_value=1e-2,
                sampling='log',
                default=1e-4
            )
        ),
        input_shape=(X_train_scaled.shape[1],)
    ))

    # ➕ Capas ocultas adicionales
    for i in range(1, num_layers):
        model.add(layers.Dense(
            units=hp.Int(
                name=f'units_{i}',
                min_value=32,
                max_value=512,
                step=32,
                default=64
            ),
            activation=hp.Choice(
                name=f'activation_{i}',
                values=['relu', 'tanh', 'sigmoid'],
                default='relu'
            ),
            kernel_regularizer=keras.regularizers.l2(
                hp.Float(
                    name=f'l2_{i}',
                    min_value=1e-5,
                    max_value=1e-2,
                    sampling='log',
                    default=1e-4
                )
            )
        ))

        # 🚫 Agregar dropout entre capas
        model.add(layers.Dropout(
            rate=hp.Float(
                name=f'dropout_{i}',
                min_value=0.0,
                max_value=0.5,
                step=0.1,
                default=0.2
            )
        ))

    # 🎯 Capa de salida para clasificación binaria
    model.add(layers.Dense(1, activation='sigmoid'))

    # ⚙️ Seleccionar optimizador
    optimizer_choice = hp.Choice(
        name='optimizer',
        values=['adam', 'sgd', 'rmsprop'],
        default='adam'
    )

    # 📐 Compilar modelo
    model.compile(
        optimizer=optimizer_choice,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model

# 🧪 Prueba de la función
print("🧪 PRUEBA DE LA FUNCIÓN BUILD_MODEL")
print("=" * 40)

# Crear un objeto de hiperparámetros de prueba
test_hp = kt.HyperParameters()
test_model = build_model(test_hp)

print("✅ Función build_model creada exitosamente")
print(f"📊 Modelo de prueba creado con arquitectura:")
test_model.summary()


# 🚀 EJERCICIO 1: Investigación e Implementación de Hyperband
# 📚 Teoría: Algoritmo Hyperband
# Hyperband es un algoritmo de optimización de hiperparámetros basado en el problema de "multi-armed bandit" que utiliza early stopping de manera principiada.
# 
# 🎯 Principios Fundamentales
# Hyperband se basa en el algoritmo Successive Halving:
# 
# R: Presupuesto máximo de recursos
# η: Factor de reducción (típicamente 3 o 4)
# r_i: Recursos asignados en la iteración i
# 🔄 Proceso de Optimización
# Inicialización: Se generan n configuraciones aleatorias
# Evaluación: Cada configuración se entrena con R/η^k recursos
# Selección: Se mantienen las mejores η configuraciones
# Iteración: Se repite el proceso aumentando los recursos
# ⚡ Ventajas Principales
# Eficiencia Computacional: Elimina configuraciones pobres rápidamente
# No requiere conocimiento previo: No necesita configuración manual
# Balanceo automático: Equilibra exploración vs explotación
# Escalabilidad: Funciona bien con espacios grandes de hiperparámetros
# ⚠️ Consideraciones Importantes
# Funciona mejor cuando hay correlación entre rendimiento temprano y final
# Puede no ser óptimo para modelos que requieren muchas épocas para converger
# El factor η debe ajustarse según el problema específico

# In[ ]:


# 🚀 Implementación de Hyperband
print("🚀 CONFIGURANDO HYPERBAND TUNER")
print("=" * 45)

# Configurar Hyperband tuner
hyperband_tuner = kt.Hyperband(
    hypermodel=build_model,  # Función que construye el modelo
    objective='val_accuracy',  # Métrica a optimizar
    max_epochs=50,  # Número máximo de épocas
    factor=3,  # Factor de reducción η
    hyperband_iterations=2,  # Número de iteraciones de Hyperband
    directory='hyperband_results',  # Directorio para guardar resultados
    project_name='breast_cancer_hyperband',  # Nombre del proyecto
    overwrite=True  # Sobrescribir resultados anteriores
)

# Mostrar información del tuner
print(f"📊 Objetivo de optimización: {hyperband_tuner.objective.name}")
print(f"🔧 Factor de reducción: {hyperband_tuner.factor}")
print(f"⏱️ Épocas máximas: {hyperband_tuner.max_epochs}")
print(f"🔄 Iteraciones de Hyperband: {hyperband_tuner.hyperband_iterations}")

# Configurar callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

print("✅ Hyperband tuner configurado exitosamente")


# In[ ]:


# 🔍 Ejecutar búsqueda con Hyperband
print("🔍 EJECUTANDO BÚSQUEDA HYPERBAND")
print("=" * 40)

import time
start_time = time.time()

# Ejecutar la búsqueda
hyperband_tuner.search(
    x=X_train_scaled,
    y=y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
    batch_size=32
)

end_time = time.time()
hyperband_duration = end_time - start_time

print(f"\n⏱️ Tiempo total de búsqueda: {hyperband_duration:.2f} segundos")
print("✅ Búsqueda Hyperband completada exitosamente")

# Obtener los mejores hiperparámetros
best_hps_hyperband = hyperband_tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"\n🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS POR HYPERBAND:")
print("=" * 55)
print(f"📊 Número de capas: {best_hps_hyperband.get('num_layers')}")
print(f"⚙️ Optimizador: {best_hps_hyperband.get('optimizer')}")

for i in range(best_hps_hyperband.get('num_layers')):
    print(f"🔸 Capa {i+1}:")
    print(f" • Unidades: {best_hps_hyperband.get(f'units_{i}')}")
    print(f" • Activación: {best_hps_hyperband.get(f'activation_{i}')}")
    print(f" • L2 regularization: {best_hps_hyperband.get(f'l2_{i}'):.2e}")
    if i > 0:
        print(f" • Dropout: {best_hps_hyperband.get(f'dropout_{i}')}")


# In[ ]:


# 📊 Análisis de resultados de Hyperband
print("📊 ANÁLISIS DE RESULTADOS - HYPERBAND")
print("=" * 42)

# Obtener todos los trials
hyperband_trials = hyperband_tuner.oracle.get_best_trials(num_trials=10)

# Crear visualización de resultados
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Evolución de scores
trial_ids = [trial.trial_id for trial in hyperband_trials]
scores = [trial.score if trial.score is not None else 0 for trial in hyperband_trials]
axes[0, 0].plot(trial_ids, scores, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_title('🚀 Hyperband: Evolución de Scores')
axes[0, 0].set_xlabel('Trial ID')
axes[0, 0].set_ylabel('Validation Accuracy')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0.85, 1.0)

# 2. Distribución de scores
axes[0, 1].hist(scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 1].axvline(max(scores), color='red', linestyle='--', label=f'Mejor: {max(scores):.4f}')
axes[0, 1].set_title('📊 Distribución de Accuracy')
axes[0, 1].set_xlabel('Validation Accuracy')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Análisis de número de capas
num_layers_list = []
scores_by_layers = []
for trial in hyperband_trials:
    if trial.score is not None:
        num_layers_list.append(trial.hyperparameters.get('num_layers'))
        scores_by_layers.append(trial.score)
axes[1, 0].scatter(num_layers_list, scores_by_layers, alpha=0.7, s=100, c='orange')
axes[1, 0].set_title('🏗️ Número de Capas vs Accuracy')
axes[1, 0].set_xlabel('Número de Capas')
axes[1, 0].set_ylabel('Validation Accuracy')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(range(1, 6))

# 4. Análisis de optimizadores
optimizers_list = []
for trial in hyperband_trials:
    if trial.score is not None:
        optimizers_list.append(trial.hyperparameters.get('optimizer'))
from collections import Counter
opt_counts = Counter(optimizers_list)
axes[1, 1].bar(opt_counts.keys(), opt_counts.values(), color=['#ff9999', '#66b3ff', '#99ff99'])
axes[1, 1].set_title('⚙️ Distribución de Optimizadores (Top 10)')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Estadísticas de rendimiento
print(f"\n📈 ESTADÍSTICAS DE RENDIMIENTO:")
print(f" • Mejor accuracy: {max(scores):.4f}")
print(f" • Accuracy promedio: {np.mean(scores):.4f}")
print(f" • Desviación estándar: {np.std(scores):.4f}")
print(f" • Número de trials exitosos: {len([s for s in scores if s > 0])}")

print(f"\n🏗️ ANÁLISIS ARQUITECTURAL:")
layers_performance = {}
for layers, score in zip(num_layers_list, scores_by_layers):
    if layers not in layers_performance:
        layers_performance[layers] = []
    layers_performance[layers].append(score)
for layers in sorted(layers_performance.keys()):
    scores_layer = layers_performance[layers]
    print(f" • {layers} capas: Promedio = {np.mean(scores_layer):.4f}, "
          f"Mejor = {max(scores_layer):.4f} ({len(scores_layer)} trials)")


# 🧠 EJERCICIO 2: Investigación e Implementación de Optimización Bayesiana
# 📚 Teoría: Optimización Bayesiana
# La Optimización Bayesiana es una técnica de optimización global que utiliza modelos probabilísticos para encontrar el óptimo de funciones costosas de evaluar.
# 
# 🧮 Componentes Fundamentales
# 1. Modelo Sustituto (Gaussian Process)
# Un Proceso Gaussiano (GP) modela la función objetivo desconocida f(x):
# 
# μ(x): Función media (típicamente 0)
# k(x, x'): Función de covarianza (kernel)
# 2. Función de Adquisición
# Determina qué punto evaluar siguiente balanceando exploración vs explotación:
# 
# f⁺: Mejor valor observado hasta ahora
# μ(x), σ(x): Media y desviación estándar del GP
# Φ, φ: CDF y PDF de la distribución normal estándar
# 🔄 Proceso Iterativo
# Inicialización: Evaluar algunos puntos aleatorios
# Ajuste del GP: Entrenar el modelo sustituto
# Optimización de adquisición: Encontrar x* que maximiza la función de adquisición
# Evaluación: Evaluar f(x*) y agregar a los datos
# Repetir: Hasta alcanzar el presupuesto o convergencia
# ✅ Ventajas sobre Métodos Tradicionales
# Eficiencia: Requiere menos evaluaciones para encontrar el óptimo
# Principiada: Usa información de evaluaciones previas de manera óptima
# Incertidumbre: Cuantifica la confianza en las predicciones
# Balance automático: Equilibra exploración y explotación naturalmente
# 💡 Kernels Comunes en GP
# RBF (Radial Basis Function): k(x,x') = σ²exp(-||x-x'||²/2l²)
# Matérn: Para funciones menos suaves
# Linear: Para relaciones lineales
# Periodic: Para patrones periódicos

# In[ ]:


# 🧠 Implementación de Optimización Bayesiana
print("🧠 CONFIGURANDO BAYESIAN OPTIMIZATION TUNER")
print("=" * 50)

# Configurar Bayesian Optimization tuner
bayesian_tuner = kt.BayesianOptimization(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=25,  # Número de trials (menor que random search)
    num_initial_points=5,  # Puntos de exploración inicial
    alpha=1e-4,  # Parámetro de regularización del GP
    beta=2.6,  # Parámetro de exploración (UCB)
    directory='bayesian_results',
    project_name='breast_cancer_bayesian',
    overwrite=True
)

print(f"📊 Objetivo de optimización: {bayesian_tuner.objective.name}")
print(f"🔬 Máximo de trials: {bayesian_tuner.max_trials}")
print(f"🎯 Puntos iniciales: {bayesian_tuner.num_initial_points}")
print(f"🔧 Alpha (regularización): {bayesian_tuner.alpha}")
print(f"🎛️ Beta (exploración): {bayesian_tuner.beta}")

print("✅ Bayesian Optimization tuner configurado exitosamente")


# In[ ]:


# 🔍 Ejecutar búsqueda con Optimización Bayesiana
print("🔍 EJECUTANDO OPTIMIZACIÓN BAYESIANA")
print("=" * 42)

import time
start_time = time.time()

# Ejecutar la búsqueda
bayesian_tuner.search(
    x=X_train_scaled,
    y=y_train,
    epochs=40,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1,
    batch_size=32
)

end_time = time.time()
bayesian_duration = end_time - start_time

print(f"\n⏱️ Tiempo total de búsqueda: {bayesian_duration:.2f} segundos")
print("✅ Optimización Bayesiana completada exitosamente")

# Obtener los mejores hiperparámetros
best_hps_bayesian = bayesian_tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"\n🏆 MEJORES HIPERPARÁMETROS - OPTIMIZACIÓN BAYESIANA:")
print("=" * 58)
print(f"📊 Número de capas: {best_hps_bayesian.get('num_layers')}")
print(f"⚙️ Optimizador: {best_hps_bayesian.get('optimizer')}")

for i in range(best_hps_bayesian.get('num_layers')):
    print(f"🔸 Capa {i+1}:")
    print(f" • Unidades: {best_hps_bayesian.get(f'units_{i}')}")
    print(f" • Activación: {best_hps_bayesian.get(f'activation_{i}')}")
    print(f" • L2 regularization: {best_hps_bayesian.get(f'l2_{i}'):.2e}")
    if i > 0:
        print(f" • Dropout: {best_hps_bayesian.get(f'dropout_{i}')}")


# In[ ]:


# 📊 Comparación entre Hyperband y Optimización Bayesiana
print("📊 COMPARACIÓN DE MÉTODOS DE OPTIMIZACIÓN")
print("=" * 45)

# Obtener trials de ambos métodos
bayesian_trials = bayesian_tuner.oracle.get_best_trials(num_trials=15)

# Preparar datos para comparación
hyperband_scores = [trial.score for trial in hyperband_trials if trial.score is not None]
bayesian_scores = [trial.score for trial in bayesian_trials if trial.score is not None]

# Crear visualización comparativa
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Comparación de distribuciones
axes[0, 0].hist(hyperband_scores, bins=10, alpha=0.7, label='Hyperband', color='lightblue')
axes[0, 0].hist(bayesian_scores, bins=10, alpha=0.7, label='Bayesian Opt.', color='lightcoral')
axes[0, 0].set_title('📊 Distribución de Scores')
axes[0, 0].set_xlabel('Validation Accuracy')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plots comparativo
data_comparison = [hyperband_scores, bayesian_scores]
axes[0, 1].boxplot(data_comparison, labels=['Hyperband', 'Bayesian Opt.'])
axes[0, 1].set_title('📦 Comparación de Rendimiento')
axes[0, 1].set_ylabel('Validation Accuracy')
axes[0, 1].grid(True, alpha=0.3)

# 3. Evolución temporal (simulada)
trials_hyperband = list(range(1, len(hyperband_scores) + 1))
trials_bayesian = list(range(1, len(bayesian_scores) + 1))
axes[0, 2].plot(trials_hyperband, hyperband_scores, 'o-', label='Hyperband', linewidth=2)
axes[0, 2].plot(trials_bayesian, bayesian_scores, 's-', label='Bayesian Opt.', linewidth=2)
axes[0, 2].set_title('⏱️ Evolución de Scores')
axes[0, 2].set_xlabel('Trial Number')
axes[0, 2].set_ylabel('Validation Accuracy')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Estadísticas de rendimiento
methods = ['Hyperband', 'Bayesian Opt.']
best_scores = [max(hyperband_scores), max(bayesian_scores)]
mean_scores = [np.mean(hyperband_scores), np.mean(bayesian_scores)]
std_scores = [np.std(hyperband_scores), np.std(bayesian_scores)]
x_pos = np.arange(len(methods))
axes[1, 0].bar(x_pos - 0.2, best_scores, 0.4, label='Mejor Score', alpha=0.8)
axes[1, 0].bar(x_pos + 0.2, mean_scores, 0.4, label='Score Promedio', alpha=0.8)
axes[1, 0].set_title('🏆 Comparación de Rendimiento')
axes[1, 0].set_ylabel('Validation Accuracy')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(methods)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Eficiencia temporal
durations = [hyperband_duration, bayesian_duration]
efficiency = [best_scores[i] / (durations[i] / 60) for i in range(2)]  # Score por minuto
axes[1, 1].bar(methods, durations, color=['lightblue', 'lightcoral'], alpha=0.7)
axes[1, 1].set_title('⏱️ Tiempo de Ejecución')
axes[1, 1].set_ylabel('Tiempo (segundos)')
axes[1, 1].grid(True, alpha=0.3)

# 6. Eficiencia (Score/Tiempo)
axes[1, 2].bar(methods, efficiency, color=['navy', 'darkred'], alpha=0.7)
axes[1, 2].set_title('⚡ Eficiencia (Score/Minuto)')
axes[1, 2].set_ylabel('Eficiencia')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Imprimir estadísticas detalladas
print(f"\n📈 ESTADÍSTICAS COMPARATIVAS:")
print("=" * 35)
print(f"🚀 HYPERBAND:")
print(f" • Mejor accuracy: {max(hyperband_scores):.4f}")
print(f" • Accuracy promedio: {np.mean(hyperband_scores):.4f} ± {np.std(hyperband_scores):.4f}")
print(f" • Tiempo total: {hyperband_duration:.1f} segundos")
print(f" • Trials exitosos: {len(hyperband_scores)}")
print(f"\n🧠 OPTIMIZACIÓN BAYESIANA:")
print(f" • Mejor accuracy: {max(bayesian_scores):.4f}")
print(f" • Accuracy promedio: {np.mean(bayesian_scores):.4f} ± {np.std(bayesian_scores):.4f}")
print(f" • Tiempo total: {bayesian_duration:.1f} segundos")
print(f" • Trials exitosos: {len(bayesian_scores)}")
print(f"\n⚡ ANÁLISIS DE EFICIENCIA:")
print(f" • Hyperband: {efficiency[0]:.6f} score/minuto")
print(f" • Bayesian Opt.: {efficiency[1]:.6f} score/minuto")
winner = "Hyperband" if max(hyperband_scores) > max(bayesian_scores) else "Optimización Bayesiana"
print(f"\n🏆 Ganador en accuracy: {winner}")


# 📈 EJERCICIO 3: Visualización Avanzada de Resultados
# La visualización de resultados es crucial para entender el comportamiento de los algoritmos de optimización y tomar decisiones informadas sobre la selección de hiperparámetros.
# 
# 🎨 Importancia de la Visualización en Optimización
# Convergencia: Observar cómo mejoran los algoritmos con el tiempo
# Exploración vs Explotación: Entender el balance de los algoritmos
# Identificación de patrones: Detectar relaciones entre hiperparámetros
# Validación de resultados: Confirmar la calidad de la optimización
# Comunicación: Presentar resultados de manera clara
