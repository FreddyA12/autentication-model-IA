# 📚 Documentación del Sistema de Reconocimiento Facial

## 🎯 Descripción General

Este sistema permite reconocer rostros de personas específicas (Alison, Freddy, Isma) y detectar personas desconocidas. Utiliza **FaceNet** para extraer características faciales y **SVM** para clasificar.

---

## 📁 Estructura del Proyecto

```
APE3/
├── dataset/
│   ├── dataset_raw/           # Imágenes originales (sin procesar)
│   │   ├── alison/
│   │   ├── freddy/
│   │   └── isma/
│   ├── dataset_clean/         # Imágenes procesadas y alineadas
│   │   ├── alison/
│   │   ├── freddy/
│   │   └── isma/
│   ├── embeddings/            # Vectores de características extraídos
│   │   └── face_embeddings.pkl
│   ├── models/                # Modelos entrenados
│   │   ├── face_svm.pkl
│   │   ├── face_embedding_classifier.keras
│   │   └── class_indices.json
│   ├── test_data/             # Imágenes para probar el sistema
│   ├── videos/                # Videos para extraer frames
│   └── scripts/               # Scripts de procesamiento
└── src/                       # Código fuente de la aplicación
```

---

## 🔄 Pipeline de Entrenamiento

### Paso 1: Preparar Videos/Imágenes

Coloca los videos o imágenes de cada persona en:
```
dataset/videos/
```

O directamente las imágenes en:
```
dataset/dataset_raw/{nombre_persona}/
```

---

### Paso 2: Extraer Frames de Videos (Opcional)

Si tienes videos, extrae los frames:

```powershell
python dataset/scripts/1_extract_frames.py
```

**¿Qué hace?**
- Lee los videos de `dataset/videos/`
- Extrae frames cada cierto intervalo
- Guarda las imágenes en `dataset/dataset_raw/{persona}/`

---

### Paso 3: Preprocesar y Alinear Rostros

```powershell
python dataset/scripts/2_preprocess_aligned.py
```

**¿Qué hace?**
1. Lee imágenes de `dataset/dataset_raw/`
2. Detecta rostros usando **MTCNN**
3. Alinea las caras usando los landmarks de los ojos
4. Extrae **embeddings de 512 dimensiones** con FaceNet
5. Guarda:
   - Imágenes alineadas en `dataset/dataset_clean/`
   - Embeddings en `dataset/embeddings/face_embeddings.pkl`

**Salida esperada:**
```
Procesando: alison (430 imágenes)
  ✓ Procesadas: 429
  ✗ Fallidas: 1

Distancias promedio entre embeddings:
  alison: 0.798 ± 0.160
  freddy: 0.653 ± 0.167
  isma: 0.665 ± 0.169

Distancias entre personas diferentes:
  alison vs freddy: 1.382
  alison vs isma: 1.122
  freddy vs isma: 1.208
```

> **Nota:** Las distancias intra-clase (~0.6-0.8) deben ser menores que las inter-clase (~1.1-1.4) para un buen reconocimiento.

---

### Paso 4: Entrenar el Clasificador

```powershell
python dataset/scripts/3_train_with_embeddings.py
```

**¿Qué hace?**
1. Carga los embeddings de `dataset/embeddings/`
2. Entrena un clasificador **SVM** (Support Vector Machine)
3. Entrena una **red neuronal** pequeña como alternativa
4. Guarda los modelos en `dataset/models/`

**Salida esperada:**
```
ENTRENANDO CLASIFICADOR SVM
  Accuracy: 100.00%
  Validación cruzada: 99.93% ± 0.15%

ENTRENANDO CLASIFICADOR NEURAL
  Accuracy: 100.00%
```

**Modelos generados:**
- `face_svm.pkl` - Clasificador SVM
- `face_embedding_classifier.keras` - Red neuronal
- `class_indices.json` - Mapeo de clases

---

### Paso 5: Probar el Sistema

```powershell
python dataset/scripts/4_predict_embeddings.py
```

**¿Qué hace?**
1. Carga los modelos entrenados
2. Prueba con imágenes de `dataset/test_data/`
3. Prueba con muestras aleatorias del dataset

**Salida esperada:**
```
PROBANDO IMÁGENES EXTERNAS
✅ alison.jpg    → alison (conf: 99.6%, dist: 0.58)
✅ freddy2.jpg   → freddy (conf: 98.8%, dist: 0.52)
✅ isma.jpg      → isma (conf: 100.0%, dist: 0.46)
⚠️  rafa.jpg     → DESCONOCIDO (conf: 67.1%, dist: 1.23)
⚠️  william.jpg  → DESCONOCIDO (conf: 44.6%, dist: 0.86)
```

---

## 📋 Resumen de Comandos

| Paso | Comando | Descripción |
|------|---------|-------------|
| 1 | `python dataset/scripts/1_extract_frames.py` | Extrae frames de videos |
| 2 | `python dataset/scripts/2_preprocess_aligned.py` | Preprocesa y extrae embeddings |
| 3 | `python dataset/scripts/3_train_with_embeddings.py` | Entrena clasificador |
| 4 | `python dataset/scripts/4_predict_embeddings.py` | Prueba el sistema |

---

## ⚙️ Configuración y Umbrales

### En `4_predict_embeddings.py`:

```python
CONFIDENCE_THRESHOLD = 0.60  # Mínima confianza para aceptar predicción
DISTANCE_THRESHOLD = 1.0     # Máxima distancia para considerar conocido
```

- Si la **confianza < 60%** → Se marca como DESCONOCIDO
- Si la **distancia > 1.0** → Se marca como DESCONOCIDO

### Ajustar umbrales:
- **Aumentar `CONFIDENCE_THRESHOLD`** → Más estricto (menos falsos positivos)
- **Disminuir `DISTANCE_THRESHOLD`** → Más estricto

---

## 🆕 Agregar una Nueva Persona

1. Crear carpeta con el nombre en `dataset/dataset_raw/`:
   ```
   dataset/dataset_raw/nueva_persona/
   ```

2. Agregar imágenes (mínimo 100, idealmente 300+)

3. Ejecutar el pipeline completo:
   ```powershell
   python dataset/scripts/2_preprocess_aligned.py
   python dataset/scripts/3_train_with_embeddings.py
   ```

4. Verificar:
   ```powershell
   python dataset/scripts/4_predict_embeddings.py
   ```

---

## 🔧 Tecnologías Utilizadas

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| Detección de rostros | MTCNN | Detecta y alinea caras |
| Extracción de features | FaceNet (InceptionResnetV1) | Genera embeddings de 512D |
| Clasificación | SVM / Red Neural | Identifica a la persona |
| Framework | TensorFlow + PyTorch | Deep Learning |

---

## 📊 Métricas de Calidad

### Distancias de Embeddings:
- **Intra-clase** (misma persona): Debe ser **< 1.0**
- **Inter-clase** (diferentes personas): Debe ser **> 1.0**
- **Ratio ideal**: Inter/Intra > 1.5

### Resultados actuales:
| Persona | Distancia Intra-clase |
|---------|----------------------|
| alison | 0.798 ± 0.160 |
| freddy | 0.653 ± 0.167 |
| isma | 0.665 ± 0.169 |

| Comparación | Distancia Inter-clase |
|-------------|----------------------|
| alison vs freddy | 1.382 |
| alison vs isma | 1.122 |
| freddy vs isma | 1.208 |

---

## ❓ Solución de Problemas

### El sistema no detecta rostros
- Verificar que las imágenes tengan buena iluminación
- Verificar que los rostros no estén muy pequeños o borrosos
- El rostro debe ocupar al menos 40x40 píxeles

### Baja precisión
- Agregar más imágenes de entrenamiento
- Asegurar variedad: diferentes ángulos, iluminación, expresiones
- Verificar que las distancias inter-clase sean mayores que intra-clase

### Muchos falsos positivos (reconoce desconocidos como conocidos)
- Aumentar `CONFIDENCE_THRESHOLD` (ej: 0.70)
- Disminuir `DISTANCE_THRESHOLD` (ej: 0.9)

### Muchos falsos negativos (no reconoce personas conocidas)
- Disminuir `CONFIDENCE_THRESHOLD` (ej: 0.50)
- Aumentar `DISTANCE_THRESHOLD` (ej: 1.2)
- Agregar más imágenes de esa persona

---

## 📝 Notas Importantes

1. **FaceNet es un modelo preentrenado** en millones de caras (VGGFace2). Solo el clasificador SVM/Neural se entrena con tus datos.

2. **Cantidad de datos recomendada:**
   - Mínimo: 100 imágenes por persona
   - Óptimo: 300-500 imágenes por persona
   - Las imágenes deben tener variedad

3. **Formato de imágenes:** JPG, PNG o JPEG

4. **Tamaño procesado:** 160x160 píxeles (automático)

---

## 🚀 Uso en Producción

Para usar el sistema en tiempo real (cámara web), ejecuta:

```powershell
python src/dual_auth/run_dual_auth_live.py
```

Esto activará la cámara y realizará reconocimiento facial en vivo.
