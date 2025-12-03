# 📚 Documentación del Sistema de Reconocimiento Facial

## 🎯 Descripción General

Sistema de reconocimiento facial basado en **FaceNet + Red Neuronal**.

**Arquitectura:**
```
imagen → MTCNN → FaceNet → embedding (512) → TU RED NEURONAL → predicción
```

- **FaceNet**: Modelo preentrenado que extrae características faciales (NO se entrena)
- **Tu Red Neuronal**: Clasificador que TÚ entrenas sobre los embeddings

---

## 📁 Estructura del Proyecto

```
APE3/
├── dataset/
│   ├── dataset_raw/           # Imágenes originales (sin procesar)
│   │   ├── alison/
│   │   ├── freddy/
│   │   └── isma/
│   ├── dataset_clean/         # Imágenes procesadas y alineadas (160x160)
│   │   ├── alison/
│   │   ├── freddy/
│   │   └── isma/
│   ├── embeddings/            # Vectores de 512 dimensiones
│   │   └── embeddings_dataset.pkl
│   ├── models/                # Modelos entrenados
│   │   ├── face_classifier.keras      # TU modelo entrenado
│   │   ├── face_classifier_best.keras # Mejor checkpoint
│   │   ├── class_indices.json         # Mapeo de clases
│   │   └── training_history.png       # Gráfica de entrenamiento
│   ├── test_data/             # Imágenes para probar
│   ├── videos/                # Videos para extraer frames
│   └── scripts/               # Scripts de procesamiento
│       ├── 1_extract_frames.py
│       ├── 2_preprocess_and_extract_embeddings.py
│       ├── 3_train_classifier.py
│       └── 4_predict.py
└── src/                       # Código de la aplicación
```

---

## 🔄 Pipeline de Entrenamiento (4 Pasos)

### Paso 1: Preparar Datos (Opcional)

Si tienes videos, extrae los frames:

```powershell
python dataset/scripts/1_extract_frames.py
```

O coloca directamente las imágenes en `dataset/dataset_raw/{nombre_persona}/`

---

### Paso 2: Preprocesar y Extraer Embeddings

```powershell
python dataset/scripts/2_preprocess_and_extract_embeddings.py
```

**¿Qué hace?**
```
imagen → MTCNN → FaceNet → embedding (512 dimensiones)
```

1. **MTCNN** detecta y ALINEA las caras usando landmarks (ojos, nariz, boca)
2. **FaceNet** convierte cada cara en un vector de 512 números
3. Guarda:
   - Imágenes alineadas en `dataset_clean/`
   - Embeddings en `dataset/embeddings/embeddings_dataset.pkl`

**¿Por qué embeddings?**
- FaceNet ya aprendió a extraer características faciales
- Dos caras de la MISMA persona → embeddings CERCANOS
- Dos caras de personas DIFERENTES → embeddings LEJANOS

**Salida esperada:**
```
X (embeddings):
    Shape: (1373, 512)  ← 1373 imágenes, 512 dimensiones cada una

y (etiquetas):
    Shape: (1373,)      ← clase de cada imagen

Distancias INTRA-clase (misma persona):
   alison: 0.798 ± 0.160
   freddy: 0.653 ± 0.167
   isma: 0.665 ± 0.169

Distancias INTER-clase (diferentes personas):
   alison vs freddy: 1.382
   alison vs isma: 1.122
   freddy vs isma: 1.208
```

---

### Paso 3: Entrenar TU Red Neuronal

```powershell
python dataset/scripts/3_train_classifier.py
```

**¿Qué hace?**

Entrena TU PROPIO MODELO sobre los embeddings.

**Arquitectura (la que usan bancos y universidades):**
```
Input (512)
    ↓
Dense(256, relu) + Dropout(0.3)
    ↓
Dense(128, relu) + Dropout(0.2)
    ↓
Dense(num_clases, softmax)
```

**¿Por qué funciona tan bien?**
- FaceNet ya hizo el trabajo duro (extraer características)
- Tu red SOLO aprende a separar las clases
- Con 400 imágenes por persona puedes lograr >99% accuracy

**Salida esperada:**
```
Accuracy final: 100.00%

Reporte de clasificación:
              precision    recall  f1-score
      alison       1.00      1.00      1.00
      freddy       1.00      1.00      1.00
        isma       1.00      1.00      1.00
```

---

### Paso 4: Predecir

```powershell
python dataset/scripts/4_predict.py
```

**Pipeline de predicción:**
```
imagen → MTCNN → FaceNet → embedding → TU MODELO → probabilidades
                                                        ↓
                                                Freddy: 92%
                                                Melanie: 7%
                                                Jose: 1%
```

**Regla de decisión:**
- Si max_prob >= 50% → ES esa persona
- Si max_prob < 50% → DESCONOCIDO

**Salida esperada:**
```
PROBANDO IMÁGENES EXTERNAS
   ✅ alison.jpg    → alison (99.6%)
   ✅ freddy2.jpg   → freddy (98.8%)
   ✅ isma.jpg      → isma (100.0%)
   ⚠️  rafa.jpg     → DESCONOCIDO (max: 67.1%)
   ⚠️  william.jpg  → DESCONOCIDO (max: 44.6%)
```

---

## 📋 Resumen de Comandos

| Paso | Comando | Descripción |
|------|---------|-------------|
| 1 | `python dataset/scripts/1_extract_frames.py` | Extrae frames de videos |
| 2 | `python dataset/scripts/2_preprocess_and_extract_embeddings.py` | Preprocesa + extrae embeddings |
| 3 | `python dataset/scripts/3_train_classifier.py` | Entrena TU red neuronal |
| 4 | `python dataset/scripts/4_predict.py` | Prueba el sistema |

---

## ⚙️ Configuración

### Umbral de confianza (en `4_predict.py`):

```python
CONFIDENCE_THRESHOLD = 0.50  # 50%
```

- **Aumentar** (ej: 0.70) → Más estricto, menos falsos positivos
- **Disminuir** (ej: 0.40) → Menos estricto, menos falsos negativos

---

## 🆕 Agregar una Nueva Persona

1. Crear carpeta:
   ```
   dataset/dataset_raw/nueva_persona/
   ```

2. Agregar imágenes (mínimo 100, idealmente 300+)

3. Ejecutar pipeline:
   ```powershell
   python dataset/scripts/2_preprocess_and_extract_embeddings.py
   python dataset/scripts/3_train_classifier.py
   ```

---

## 🔧 Tecnologías

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| Detección | MTCNN | Detecta y alinea caras |
| Embeddings | FaceNet (InceptionResnetV1) | Extrae vectores de 512D |
| Clasificación | Tu Red Neuronal | Identifica personas |
| Framework | TensorFlow + PyTorch | Deep Learning |

---

## ❓ Solución de Problemas

### Baja precisión
- Agregar más imágenes variadas (ángulos, luz, expresiones)
- Verificar que distancias inter-clase > intra-clase

### Muchos "DESCONOCIDO"
- Disminuir `CONFIDENCE_THRESHOLD` (ej: 0.40)

### Falsos positivos (reconoce desconocidos)
- Aumentar `CONFIDENCE_THRESHOLD` (ej: 0.70)
