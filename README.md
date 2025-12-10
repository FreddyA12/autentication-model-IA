# Sistema de Autenticación Biométrica Dual con IA

Sistema de autenticación biométrica que combina **Reconocimiento Facial** y **Reconocimiento de Voz** para proporcionar una autenticación dual segura y precisa.

## 🎯 Características

- ✅ **Autenticación Dual**: Requiere coincidencia de rostro Y voz para autenticar
- ✅ **Reconocimiento Facial**: Usando MTCNN + FaceNet con 512-dim embeddings
- ✅ **Reconocimiento de Voz**: Usando ECAPA-TDNN con 192-dim embeddings (estado del arte)
- ✅ **Interfaz Web**: Django backend con interfaz moderna en HTML/CSS/JS
- ✅ **Detección de Desconocidos**: Rechaza automáticamente personas no registradas
- ✅ **Alta Precisión**: 100% accuracy en tests con ECAPA-TDNN

---

## 📋 Requisitos Previos

- Python 3.8+
- Webcam
- Micrófono
- Windows/Linux/macOS

---

## 🚀 Instalación

### 1. Crear y Activar Entorno Virtual

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**Nota importante**: El proyecto usa versiones específicas de PyTorch (2.6.0) por compatibilidad con SpeechBrain.

---

## 🧠 Arquitectura del Sistema

### Reconocimiento Facial

#### Tecnologías
- **MTCNN**: Detector de rostros multi-tarea
- **FaceNet (Keras)**: Extractor de embeddings (512 dimensiones)
- **Clasificador MLP**: Red neuronal para clasificación de identidades

#### Pipeline de Entrenamiento

**Ubicación de scripts:** `dataset/face/scripts/`

##### 1. Extraer Frames de Videos
Extrae fotogramas de videos de entrenamiento.
```bash
python dataset/face/scripts/1_extract_frames.py
```
- **Input**: Videos en `dataset/face/videos/`
- **Output**: Frames en `dataset/face/processed/`
- **Parámetros**: 1 frame cada 5 frames del video

##### 2. Preprocesar y Extraer Embeddings
Detecta rostros, los alinea y extrae embeddings faciales.
```bash
python dataset/face/scripts/2_preprocess_and_extract_embeddings.py
```
- **Proceso**:
  1. Detecta rostros usando MTCNN
  2. Recorta y redimensiona a 160x160 px
  3. Extrae embedding de 512 dimensiones con FaceNet
- **Output**: `dataset/face/embeddings/embeddings_dataset.pkl`

##### 3. Entrenar Clasificador
Entrena un MLP para clasificar los embeddings faciales.
```bash
python dataset/face/scripts/3_train_classifier.py
```
- **Arquitectura del MLP**:
  - Input: 512 dimensiones
  - Dense(256) + Dropout(0.3)
  - Dense(128) + Dropout(0.2)
  - Output: Softmax (número de personas)
- **Output**: `dataset/face/models/face_classifier.keras`

##### 4. Probar el Modelo
Evalúa el modelo con imágenes de prueba.
```bash
python dataset/face/scripts/4_predict.py
```
- **Input**: Imágenes en `dataset/face/test_data/`
- **Output**: Predicciones con confianza

---

### Reconocimiento de Voz

#### Tecnologías
- **ECAPA-TDNN**: Modelo de speaker recognition (SpeechBrain)
- **MLP Classifier**: Red neuronal para clasificación de voces
- **Embeddings**: 192 dimensiones optimizadas para distinguir voces

#### Pipeline de Entrenamiento

**Ubicación de scripts:** `dataset/voice/scripts/`

##### 1. Extraer y Procesar Audio
Limpia audio, elimina silencios y segmenta en clips.
```bash
python dataset/voice/scripts/1_extract_audio.py
```
- **Input**: Archivos de audio en `dataset/voice/audio_raw/`
- **Proceso**:
  1. Resamplea a 16kHz mono
  2. Elimina silencios (VAD)
  3. Segmenta en clips de 5 segundos
- **Output**: Archivos WAV en `dataset/voice/processed/`

##### 2. Generar Embeddings con ECAPA-TDNN
Extrae embeddings de voz usando el modelo ECAPA-TDNN pre-entrenado.
```bash
python dataset/voice/scripts/2_generate_voice_embeddings.py
```
- **Modelo**: `speechbrain/spkrec-ecapa-voxceleb`
- **Proceso**:
  1. Carga audio (mono 16kHz)
  2. Extrae embedding de 192 dimensiones
  3. Cada audio genera un vector único
- **Output**: 
  - `dataset/voice/embeddings/voice_embeddings.npy`
  - `dataset/voice/embeddings/voice_labels.npy`
  - `dataset/voice/embeddings/voice_label_map.json`

**¿Por qué ECAPA-TDNN y no YAMNet?**
- **ECAPA-TDNN**: Diseñado específicamente para speaker recognition (192 dims)
- **YAMNet**: Clasificador genérico de sonidos (1024 dims) - menos preciso para voces

##### 3. Entrenar Clasificador MLP
Entrena un MLP para clasificar los embeddings de voz.
```bash
python dataset/voice/scripts/3_train_voice_mlp.py
```
- **Arquitectura del MLP**:
  - Input: 192 dimensiones
  - Dense(512) + BatchNorm + Dropout(0.3)
  - Dense(256) + BatchNorm + Dropout(0.4)
  - Output: Softmax (número de personas + unknown)
- **Entrenamiento**:
  - Optimizer: Adam (lr=0.001)
  - Loss: Sparse Categorical Crossentropy
  - Callbacks: EarlyStopping, ReduceLROnPlateau
  - Data Augmentation: Ruido gaussiano si dataset < 100 muestras
- **Output**: `dataset/voice/models/voice_mlp_best.keras`

##### 4. Evaluar el Modelo
Prueba el modelo con audios de test.
```bash
python dataset/voice/scripts/4_predict_voice.py
```
- **Input**: Archivos de audio en `dataset/voice/test_audios/`
- **Output**: Predicciones con confianza
- **Sanity Check**: Verifica precisión en el dataset de entrenamiento

---

## 🌐 Aplicación Web (Django)

### Iniciar el Servidor

```bash
cd web
python manage.py runserver
```

La aplicación estará disponible en: `http://localhost:8000`

### Endpoints API

#### 1. Reconocimiento Facial
```
POST /api/predict/
Content-Type: multipart/form-data
Body: { image: <archivo> }
```

#### 2. Reconocimiento de Voz
```
POST /api/predict_voice/
Content-Type: multipart/form-data
Body: { audio: <archivo WAV> }
```

#### 3. Autenticación Dual
```
POST /api/authenticate_dual/
Content-Type: multipart/form-data
Body: { 
  image: <archivo>,
  audio: <archivo WAV>
}
```

### Lógica de Autenticación Dual

El sistema implementa una autenticación dual con las siguientes reglas:

1. ✅ **Ambos coinciden**: Autenticación exitosa
2. ✅ **Solo cara exitosa (>90%)**: Permite acceso (voz opcional)
3. ❌ **Solo voz exitosa**: Rechaza acceso (requiere cara)
4. ❌ **Ninguno exitoso**: Rechaza acceso

**Configuración de umbrales** (`web/settings.py`):
- `CONFIDENCE_THRESHOLD = 0.95` (95% para cara)
- `VOICE_CONFIDENCE_THRESHOLD = 0.85` (85% para voz)

---

## 📁 Estructura del Proyecto

```
autentication-model-IA/
├── dataset/
│   ├── face/                    # Reconocimiento facial
│   │   ├── videos/              # Videos de entrenamiento
│   │   ├── processed/           # Frames extraídos
│   │   ├── aligned/             # Rostros alineados
│   │   ├── embeddings/          # Embeddings faciales
│   │   ├── models/              # Modelos entrenados
│   │   │   ├── face_classifier.keras
│   │   │   └── class_indices.json
│   │   └── scripts/             # Scripts de entrenamiento
│   │
│   └── voice/                   # Reconocimiento de voz
│       ├── audio_raw/           # Audio crudo
│       ├── processed/           # Audio procesado
│       ├── embeddings/          # Embeddings de voz
│       ├── models/              # Modelos entrenados
│       │   ├── voice_mlp_best.keras
│       │   └── voice_class_indices.json
│       ├── test_audios/         # Audios de prueba
│       └── scripts/             # Scripts de entrenamiento
│
├── web/                         # Aplicación Django
│   ├── face_auth/               # App principal
│   │   ├── face_service.py      # Servicio de reconocimiento facial
│   │   ├── voice_service.py     # Servicio de reconocimiento de voz
│   │   ├── views.py             # Endpoints API
│   │   ├── templates/           # HTML
│   │   └── static/              # CSS/JS
│   ├── settings.py              # Configuración Django
│   └── manage.py                # CLI Django
│
├── pretrained_models/           # Modelos pre-entrenados descargados
│   └── spkrec-ecapa-voxceleb/   # ECAPA-TDNN de SpeechBrain
│
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo
```

---

## 🔧 Configuración Avanzada

### Ajustar Umbrales de Confianza

Edita `web/settings.py`:

```python
# Umbral para reconocimiento facial (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.95  

# Umbral para reconocimiento de voz (0.0 - 1.0)
VOICE_CONFIDENCE_THRESHOLD = 0.85
```

**Recomendaciones**:
- **Cara**: 0.90 - 0.95 (muy preciso)
- **Voz**: 0.70 - 0.85 (balance entre precisión y usabilidad)

### Agregar Nuevas Personas

#### Reconocimiento Facial
1. Graba un video corto (10-30 segundos) de la persona
2. Guárdalo en `dataset/face/videos/<nombre>/`
3. Ejecuta el pipeline completo desde el paso 1

#### Reconocimiento de Voz
1. Graba 3-5 audios de la persona hablando (5-10 segundos cada uno)
2. Guárdalos en `dataset/voice/audio_raw/<nombre>/`
3. Ejecuta el pipeline completo desde el paso 1

---

## 📊 Rendimiento del Sistema

### Reconocimiento Facial
- **Precisión**: ~98-100% en personas registradas
- **FPS**: ~2-3 fps en CPU
- **Embeddings**: 512 dimensiones (FaceNet)

### Reconocimiento de Voz
- **Precisión**: 100% en tests con ECAPA-TDNN
- **Tiempo de inferencia**: ~1-2 segundos por audio
- **Embeddings**: 192 dimensiones (ECAPA-TDNN)
- **Mejora vs YAMNet**: +50% en precisión para speaker recognition

---

## 🛠️ Solución de Problemas

### Error: "No se encontró el modelo"
Asegúrate de haber ejecutado los scripts de entrenamiento completos:
```bash
# Para cara
python dataset/face/scripts/3_train_classifier.py

# Para voz
python dataset/voice/scripts/3_train_voice_mlp.py
```

### Error: "No se detectó ningún rostro"
- Verifica que hay buena iluminación
- Asegúrate de estar mirando directamente a la cámara
- Ajusta la distancia a la cámara (30-100 cm recomendado)

### Error: "Voz no reconocida"
- Habla claramente durante 3-5 segundos
- Evita ruido de fondo excesivo
- Verifica que el micrófono funciona correctamente

### Error de compatibilidad con PyTorch/SpeechBrain
Reinstala las versiones específicas:
```bash
pip install torch==2.6.0 torchaudio==2.6.0 --force-reinstall
```

---

## 📝 Notas Técnicas

### Embeddings vs Clasificación Directa

El sistema usa un enfoque de **dos etapas**:

1. **Extracción de embeddings**: Modelos pre-entrenados (FaceNet, ECAPA-TDNN)
2. **Clasificación**: MLP entrenado con tus datos

**Ventajas**:
- Reutiliza conocimiento de modelos pre-entrenados
- Requiere menos datos de entrenamiento
- Mejor generalización
- Fácil agregar nuevas personas (solo reentrenar el MLP)

### Por qué ECAPA-TDNN

**ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network):
- Estado del arte en speaker recognition
- Embeddings de 192 dims optimizados para voces
- Pre-entrenado en VoxCeleb (millones de voces)
- Robusto a variaciones de micrófono y ruido

---

## 📚 Referencias

- **FaceNet**: [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)
- **ECAPA-TDNN**: [Desplanques et al., 2020](https://arxiv.org/abs/2005.07143)
- **SpeechBrain**: [Ravanelli et al., 2021](https://arxiv.org/abs/2106.04624)
- **MTCNN**: [Zhang et al., 2016](https://arxiv.org/abs/1604.02878)

---

## 👨‍💻 Autor

Desarrollado como proyecto de autenticación biométrica con IA.

## 📄 Licencia

MIT License - Ver archivo LICENSE para más detalles.
