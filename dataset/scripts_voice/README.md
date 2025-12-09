# 🎤 PIPELINE DE VOZ - README

## 📋 Estructura del Pipeline

Este pipeline es **idéntico** al de reconocimiento facial, pero para voz:

```
ROSTROS:                           VOZ:
────────────────────────────────────────────────────────────
Videos                             Videos
  ↓                                  ↓
1_extract_frames.py                1_extract_audio.py
  ↓                                  ↓
Frames                             Audios WAV (16kHz)
  ↓                                  ↓
2_preprocess_embeddings.py         2_generate_voice_embeddings.py
  ↓                                  ↓
FaceNet (embeddings 512D)          ECAPA-TDNN (embeddings 192D)
  ↓                                  ↓
3_train_classifier.py              3_train_voice_mlp.py
  ↓                                  ↓
Tu CNN clasificador                Tu MLP clasificador
  ↓                                  ↓
4_predict.py                       4_predict_voice.py
```

## 🚀 Pasos para Entrenar

### Paso 0: Instalar Dependencias

```powershell
pip install -r requirements.txt
```

### Paso 1: Preparar Videos

Coloca los videos en `dataset/videos/`:

```
dataset/videos/
    freddy.mp4
    melanie.mp4
    rafael.mp4
    william.mp4
    ismael.mp4
```

**Requisitos:**
- Nombre del video = nombre de la persona
- Debe tener audio claro (2-5 segundos mínimo)

### Paso 2: Extraer Audio

```powershell
python dataset/scripts_voice/1_extract_audio.py
```

**Output:**
```
dataset/dataset_voice/
    freddy/
        freddy_001.wav (3s, 16kHz, mono)
        freddy_002.wav
        freddy_003.wav
    melanie/
        melanie_001.wav
    ...
```

### Paso 3: Generar Embeddings con ECAPA-TDNN

```powershell
python dataset/scripts_voice/2_generate_voice_embeddings.py
```

**¿Qué hace?**
- Usa **ECAPA-TDNN preentrenado** (NO lo entrenas)
- Convierte cada audio → vector de 192 números
- Guarda embeddings en `dataset/embeddings/`

**Output:**
```
dataset/embeddings/
    voice_embeddings.npy       # (N, 192)
    voice_labels.npy           # (N,)
    voice_label_map.json       # {0: 'freddy', 1: 'melanie', ...}
```

### Paso 4: Entrenar TU MLP

```powershell
python dataset/scripts_voice/3_train_voice_mlp.py
```

**¿Qué hace?**
- Entrena **TU PROPIA** red neuronal
- Arquitectura: Dense(256) → Dropout → Dense(128) → Dropout → Softmax
- Usa epochs, backpropagation, etc.
- Guarda modelo en `dataset/models/voice_mlp_best.keras`

**Esto SÍ es entrenamiento supervisado con tu dataset.**

### Paso 5: Probar el Modelo

```powershell
python dataset/scripts_voice/4_predict_voice.py <audio.wav>
```

**Ejemplos:**
```powershell
# Probar con audio del dataset
python dataset/scripts_voice/4_predict_voice.py dataset/dataset_voice/freddy/freddy_001.wav

# Probar con audio nuevo
python dataset/scripts_voice/4_predict_voice.py test_audio.wav
```

## 🧠 ¿Qué se Entrena y Qué No?

### ❌ NO entrenas ECAPA-TDNN
- Es un modelo preentrenado (como FaceNet)
- Ya sabe extraer características de voz
- Solo lo usas para obtener embeddings

### ✅ SÍ entrenas el MLP
- Es **TU modelo**
- Lo entrenas desde cero con tus datos
- Aprende a clasificar los embeddings
- Tiene epochs, loss, accuracy, etc.

## 📊 Pipeline Completo

```
Audio → ECAPA-TDNN → Embedding(192) → MLP → Identidad
        (preentrenado)                (TU modelo)
```

## 🔧 Archivos Clave

### Scripts
- `scripts_voice/1_extract_audio.py` - Extrae audio de videos
- `scripts_voice/2_generate_voice_embeddings.py` - ECAPA-TDNN embeddings
- `scripts_voice/3_train_voice_mlp.py` - Entrena tu MLP
- `scripts_voice/4_predict_voice.py` - Predice identidad

### Datos Generados
- `dataset/dataset_voice/` - Audios organizados por persona
- `dataset/embeddings/voice_*.npy` - Embeddings y labels
- `dataset/models/voice_mlp_best.keras` - Tu modelo entrenado

## 💡 Tips

1. **Mínimo de datos:**
   - 3-8 audios por persona
   - 2-5 segundos cada audio

2. **Calidad del audio:**
   - Sin ruido de fondo
   - Voz clara
   - 16 kHz (se hace automáticamente)

3. **Troubleshooting:**
   - Si el modelo tiene baja accuracy → más datos
   - Si hay overfitting → más dropout
   - Si underfitting → más epochs o neuronas

## 🎯 Resultado Esperado

```
📊 RESULTADOS DE PREDICCIÓN
================================================================

🎤 Audio: freddy_test.wav

📈 Probabilidades por clase:
   freddy        95.32%  ███████████████████████████████████████████████
   melanie        3.21%  ██
   rafael         1.23%  █
   william        0.18%  
   ismael         0.06%  

================================================================
✅ IDENTIDAD: FREDDY
   Confianza: 95.32%
================================================================
```

## ⚖️ Comparación con Rostros

| Característica | Rostros | Voz |
|---------------|---------|-----|
| **Extractor** | FaceNet (512D) | ECAPA-TDNN (192D) |
| **Clasificador** | CNN Dense | MLP Dense |
| **Input** | Frames 160x160 | Audio 16kHz |
| **Output** | Identidad + confianza | Identidad + confianza |
| **¿Se entrena extractor?** | ❌ No | ❌ No |
| **¿Se entrena clasificador?** | ✅ Sí | ✅ Sí |

---

**¡Listo!** Ahora tienes el pipeline de voz funcionando igual que el de rostros 🎉
