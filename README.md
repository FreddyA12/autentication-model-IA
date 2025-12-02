## Sistema de Autenticación Dual (Rostro + Voz)

El repositorio contiene un flujo completo para verificar identidades usando simultáneamente los rasgos faciales y de voz de cada persona. Ambos modelos se entrenan a partir de los videos del directorio `dataset/videos`, se testean por separado y, finalmente, trabajan de manera conjunta en tiempo (casi) real.

### 1. Pipeline de reconocimiento facial
1. `python dataset/scripts/1_extract_frames.py`  
   Extrae frames de cada video original y los guarda en `dataset/dataset_raw/<usuario>/`.
2. `python dataset/scripts/2_balance_and_preprocess.py`  
   Localiza el rostro con MTCNN, recorta, normaliza y deja los datos en `dataset/dataset_clean/<usuario>/`.
3. `python dataset/scripts/3_train_model.py`  
   Entrena un CNN simple con TensorFlow y guarda los modelos (`faces_cnn.keras` y `faces_cnn_best.keras`) junto con el mapeo de clases en `dataset/models/`.
4. `python dataset/scripts/4_predict.py`  
   Permite testear imágenes nuevas (`dataset/test_data/test_data`) y muestras internas del dataset limpio.

Durante la inferencia en vivo se usa la clase `FaceRecognizer` (`src/dual_auth/face_recognition.py`), que encapsula la detección con MTCNN y la clasificación con el modelo entrenado.

### 2. Pipeline de reconocimiento de voz
1. `python dataset/scripts/5_extract_audio.py`  
   Extrae la pista de audio de cada video y la guarda como WAV en `dataset/audio_raw/`.
2. `python dataset/scripts/6_build_mel_spectrograms.py`  
   Parte cada audio en segmentos de 2.5 s, genera espectrogramas log-mel y exporta cada uno como imagen (`dataset/audio_mels/<usuario>/`).
3. `python dataset/scripts/7_train_voice_model.py`  
   Entrena un CNN con imágenes de espectrogramas en escala de grises, y guarda los artefactos (`voice_cnn.keras`, `voice_cnn_best.keras`, `voice_class_indices.json`) en `dataset/models/`.
4. `python dataset/scripts/8_predict_voice.py`  
   Evalúa audios de prueba (`dataset/test_data/audio_samples/`) o, si no existen, usa los WAV generados en `dataset/audio_raw/`.

La clase `VoiceRecognizer` (`src/dual_auth/voice_recognition.py`) convierte cualquier señal de audio cruda en un espectrograma log-mel consistente con el entrenamiento y recupera la predicción del modelo.

### 3. Integración y decisión conjunta
- `src/dual_auth/dual_authenticator.py` define la lógica de fusión. Solo devuelve **Autenticado** si ambos modelos reconocen a la misma persona (coincide con `expected_identity`) y lo hacen con confianza por encima de los umbrales configurables.
- `src/dual_auth/run_dual_auth_live.py` es una demo en tiempo real que abre la cámara, graba audio de 2.5 s cada varios segundos y llama a `DualAuthenticator`.

Ejemplo de ejecución:

```bash
python -m src.dual_auth.run_dual_auth_live \
    --identity freddy \
    --face-threshold 0.75 \
    --voice-threshold 0.65
```

Durante la demo, la ventana mostrará el estado actual (grabando, autenticado o rechazado). Se puede salir con `Q`. Antes de ejecutarla asegúrate de haber completado los pasos de entrenamiento de rostro y voz para que existan los modelos en `dataset/models/`.

### 4. Dependencias
Instala las dependencias listadas en `requirements.txt`. Asegúrate de que tu entorno tenga acceso a una GPU o CPU suficientemente potente para entrenar ambos modelos, y cuenta con cámara web + micrófono para la prueba en vivo.

### 5. Estructura resultante
- `dataset/dataset_*`: datos de imágenes (raw, clean, balanced).
- `dataset/audio_raw` / `dataset/audio_mels`: datos para el modelo de voz.
- `dataset/models`: artefactos de entrenamiento (rostro y voz) y métricas.
- `dataset/scripts`: scripts numerados para reproducir cada paso.
- `src/dual_auth`: clases reutilizables y el demo de autenticación dual.

### 6. Manejo de desconocidos y dataset mínimo
- Las clases `FaceRecognizer` y `VoiceRecognizer` etiquetan como **"Desconocido"** cualquier predicción cuya confianza esté por debajo de los umbrales configurables (`recognition_threshold`). Ajusta esos valores si notas falsos positivos o negativos.
- Si solo tienes videos de dos personas conocidas, sube esos archivos a `dataset/videos/<nombre>.mp4` y repite los pasos 1‑8: se generarán suficientes muestras para entrenar ambos modelos.
- Para mejorar la detección de desconocidos puedes grabar voces o rostros adicionales (personas fuera de la lista autorizada) y guardarlos como una tercera clase; así el modelo aprenderá mejor los límites entre identidades.
- Al ejecutar `run_dual_auth_live` puedes reforzar los filtros con `--face-threshold` y `--voice-threshold`, que deben ser ≤ a los `recognition_threshold` usados al instanciar los reconocedores. Valores altos reducen falsos positivos, valores bajos evitan rechazos injustificados.

Siguiendo la secuencia anterior puedes capturar, entrenar y probar ambos modelos, y finalmente ejecutar el flujo combinado que responde únicamente con **es la persona correcta** o **no lo es**. Cuando una señal no supera la confianza mínima verás "Desconocido" tanto en las pruebas offline como en la demo en vivo.
