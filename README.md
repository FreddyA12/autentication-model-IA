# Modelo de Autenticación con IA

Este proyecto implementa un sistema de autenticación biométrica utilizando Reconocimiento Facial y Reconocimiento de Voz.

## 1. Configuración

Sigue estos pasos para preparar el entorno.

### Crear y Activar Entorno Virtual

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Instalar Dependencias

```bash
pip install -r requirements.txt
```

---

## 2. Pipeline de Reconocimiento Facial

El sistema de reconocimiento facial utiliza MTCNN para la detección y FaceNet (InceptionResnetV1) para la extracción de embeddings.

**Ubicación:** `dataset/face/scripts/`

### Paso 1: Extraer Frames de Videos
Extrae imágenes de los videos ubicados en `dataset/face/videos/`.
```bash
python dataset/face/scripts/1_extract_frames.py
```

### Paso 2: Preprocesar y Extraer Embeddings
Detecta rostros, los alinea y extrae embeddings de 512 dimensiones.
```bash
python dataset/face/scripts/2_preprocess_and_extract_embeddings.py
```

### Paso 3: Entrenar Clasificador
Entrena una red neuronal para clasificar los embeddings faciales.
```bash
python dataset/face/scripts/3_train_classifier.py
```

### Paso 4: Predecir
Prueba el modelo con imágenes de `dataset/face/test_data/` y del propio dataset.
```bash
python dataset/face/scripts/4_predict.py
```

---

## 3. Pipeline de Reconocimiento de Voz

El sistema de reconocimiento de voz utiliza YAMNet para la extracción de embeddings y un MLP para la clasificación.

**Ubicación:** `dataset/voice/scripts/`

### Paso 1: Extraer y Procesar Audio
Procesa el audio crudo de `dataset/voice/audio_raw/`, elimina silencios y lo segmenta.
```bash
python dataset/voice/scripts/1_extract_audio.py
```

### Paso 2: Generar Embeddings
Extrae embeddings de 1024 dimensiones usando YAMNet. Incluye aumento de datos (ruido, cambio de tono) para mejorar la robustez.
```bash
python dataset/voice/scripts/2_generate_voice_embeddings.py
```

### Paso 3: Entrenar MLP
Entrena un Perceptrón Multicapa (MLP) con los embeddings de voz.
```bash
python dataset/voice/scripts/3_train_voice_mlp.py
```

### Paso 4: Predecir (Archivo)
Predice la identidad de un hablante a partir de un archivo de audio (.wav, .opus, .ogg, etc.).
```bash
python dataset/voice/scripts/4_predict_voice.py
```

### Paso 5: Reconocimiento en Vivo (Micrófono)
Ejecuta el reconocimiento de voz en tiempo real usando el micrófono.
```bash
python dataset/voice/scripts/5_live_voice_recognition.py
```
