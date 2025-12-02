# 🎯 Mejoras Sugeridas para el Modelo de Reconocimiento Facial

## Estado Actual
- ✅ Modelo con transfer learning (MobileNetV2)
- ✅ Accuracy: 85.4% en validación
- ✅ Detección de desconocidos implementada
- ⚠️ Confusión entre alison/freddy/isma

## 📋 Próximos Pasos para Mejorar

### 1. **Balance de Dataset** (PRIORIDAD ALTA)
El modelo está sesgado hacia "isma" porque tiene más imágenes.

**Solución:**
```python
# Balancear el número de imágenes por persona
# Target: ~430 imágenes por cada persona (el mínimo actual)
```

**Acción:**
- Eliminar imágenes de isma hasta tener ~430
- O agregar más imágenes de alison y freddy

### 2. **Agregar Más Datos de Entrenamiento**
Solo tienes 1374 imágenes totales (458 por persona). Para mejor reconocimiento necesitas:

- **Mínimo recomendado**: 500-1000 imágenes por persona
- **Ideal**: 1000-2000 imágenes por persona

**Cómo obtener más datos:**
- Grabar videos más largos de cada persona
- Diferentes condiciones de iluminación
- Diferentes ángulos de cámara
- Diferentes expresiones faciales
- Con/sin accesorios (lentes, gorra, etc.)

### 3. **Agregar Cuarta Persona**
Mencionaste que quieres 4 rostros. Necesitas:

1. Crear video de la 4ta persona
2. Guardarlo como `dataset/videos/nombre.mp4`
3. Ejecutar scripts 1-3 nuevamente

### 4. **Clase "Unknown" Explícita** (AVANZADO)
En lugar de usar solo umbral de confianza, entrenar con una clase "unknown":

**Pasos:**
1. Recolectar 500+ imágenes de rostros desconocidos (de internet/otros videos)
2. Crear carpeta `dataset/dataset_clean/unknown/`
3. Reentrenar el modelo con 4 clases

### 5. **Fine-tuning del Modelo Base**
Actualmente MobileNetV2 está congelado. Para mejor rendimiento:

```python
# Después de entrenar, descongelar las últimas capas
base_model.trainable = True
# Congelar solo las primeras capas
for layer in base_model.layers[:100]:
    layer.trainable = False
# Reentrenar con learning rate muy bajo
```

### 6. **Aumentar Umbral de Confianza Gradualmente**
- Actual: 50%
- Recomendado: Probar 55%, 60%, 65% según resultados
- Usar validación cruzada para encontrar el óptimo

### 7. **Agregar Más Validación**
Crear un conjunto de test separado con:
- Imágenes nuevas de cada persona
- Imágenes de desconocidos
- Diferentes condiciones (distancia, ángulo, luz)

## 🚀 Plan de Acción Rápido

### Paso 1: Balance Inmediato
```bash
# Ejecuta esto para balancear:
python dataset/scripts/balance_dataset.py
```

### Paso 2: Re-entrenar
```bash
python dataset/scripts/3_train_model.py
```

### Paso 3: Validar
```bash
python dataset/scripts/4_predict.py
```

## 📊 Métricas a Monitorear

- **Accuracy por clase**: Debe ser similar (~80-90%) para todas
- **Matriz de confusión**: Revisar qué clases se confunden
- **False Positive Rate**: Personas desconocidas aceptadas
- **False Negative Rate**: Personas conocidas rechazadas

## 🎓 Conceptos Clave

### Sobre el Umbral de Confianza:
- **Muy bajo (30-40%)**: Acepta muchos falsos positivos (desconocidos como conocidos)
- **Medio (50-60%)**: Balance entre precisión y recall
- **Alto (70-80%)**: Rechaza incluso personas legítimas

### Sobre el Data Augmentation:
Ayuda al modelo a generalizar mejor con variaciones de:
- Rotación
- Zoom
- Brillo
- Contraste
- Pero NO reemplaza tener más datos reales

## ⚙️ Script de Balance (Crear)

Crea `dataset/scripts/balance_dataset.py`:

```python
import os
import shutil
import random

CLEAN_DIR = "dataset/dataset_clean"
TARGET_COUNT = 430  # Mínimo actual

for person in os.listdir(CLEAN_DIR):
    person_dir = os.path.join(CLEAN_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    
    images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png'))]
    
    if len(images) > TARGET_COUNT:
        # Eliminar imágenes aleatorias
        to_remove = random.sample(images, len(images) - TARGET_COUNT)
        for img in to_remove:
            os.remove(os.path.join(person_dir, img))
        print(f"{person}: Reducido de {len(images)} a {TARGET_COUNT}")
    else:
        print(f"{person}: OK con {len(images)} imágenes")
```

## 📝 Notas Finales

1. **El modelo actual (85.4% accuracy) es BUENO pero no perfecto**
2. **La confusión entre personas es normal con pocos datos**
3. **El transfer learning está ayudando mucho**
4. **Necesitas más datos para llegar a >95% accuracy**
5. **El umbral de confianza es crucial para detectar desconocidos**

## 🎯 Objetivo Realista

Con los datos actuales:
- **Mejor caso esperado**: 88-92% accuracy
- **Para 95%+**: Necesitas 2-3x más datos por persona
- **Para producción**: Necesitas validación exhaustiva con casos reales
