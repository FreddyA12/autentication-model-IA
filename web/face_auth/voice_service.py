"""
Voice Recognition Service

Servicio para reconocer identidad por voz usando ECAPA-TDNN + MLP
"""

import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
import tensorflow as tf
from django.conf import settings


class VoiceRecognitionService:
    """
    Singleton service para reconocimiento de voz
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoiceRecognitionService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # No inicializar en __init__, solo en lazy_init()
        pass
    
    def lazy_init(self):
        """Inicialización perezosa cuando Django settings esté disponible"""
        if not self._initialized:
            print("🎤 Inicializando Voice Recognition Service...")
            
            # Configuración
            self.models_dir = Path(settings.PROJECT_ROOT) / "dataset" / "models"
            self.model_path = self.models_dir / "voice_mlp_best.keras"
            self.class_indices_path = self.models_dir / "voice_class_indices.json"
            self.confidence_threshold = getattr(settings, 'VOICE_CONFIDENCE_THRESHOLD', 0.80)  # Subido a 0.80 para mejor detección de desconocidos
            
            # Verificar que existan los modelos
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modelo de voz no encontrado: {self.model_path}")
            
            if not self.class_indices_path.exists():
                raise FileNotFoundError(f"Mapeo de clases no encontrado: {self.class_indices_path}")
            
            # Cargar ECAPA-TDNN
            print("   Cargando ECAPA-TDNN...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            
            from speechbrain.utils.fetching import LocalStrategy
            
            self.ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": device},
                local_strategy=LocalStrategy.COPY
            )
            
            # Cargar mapeo de clases PRIMERO
            print("   Cargando mapeo de clases...")
            import json
            with open(self.class_indices_path, 'r', encoding='utf-8') as f:
                self.label_map = json.load(f)
                self.label_map = {int(k): v for k, v in self.label_map.items()}
            
            # Cargar MLP
            print("   Cargando MLP...")
            try:
                # Intentar cargar con compile=False
                self.mlp = tf.keras.models.load_model(
                    str(self.model_path),
                    compile=False
                )
            except Exception as e:
                print(f"   ⚠️  Error al cargar modelo: {e}")
                print(f"   Intentando rebuild del modelo...")
                # Si falla, reconstruir el modelo manualmente
                self.mlp = self._build_mlp()
                print(f"   ✅ Modelo reconstruido exitosamente")
            
            print(f"   ✅ Servicio listo ({len(self.label_map)} clases)")
            print(f"   Dispositivo: {device}")
            print(f"   Umbral de confianza: {self.confidence_threshold}")
            
            self._initialized = True
    
    def _build_mlp(self):
        """Reconstruir el modelo MLP con arquitectura actualizada"""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        num_classes = len(self.label_map)
        
        model = keras.Sequential([
            layers.Input(shape=(192,)),
            
            # Arquitectura actualizada (debe coincidir con el entrenamiento)
            layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense1'),
            layers.BatchNormalization(name='bn1'),
            layers.Dropout(0.5, name='dropout1'),
            
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense2'),
            layers.BatchNormalization(name='bn2'),
            layers.Dropout(0.5, name='dropout2'),
            
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense3'),
            layers.BatchNormalization(name='bn3'),
            layers.Dropout(0.3, name='dropout3'),
            
            layers.Dense(num_classes, activation='softmax', name='output')
        ], name='VoiceMLP')
        
        # Cargar pesos desde el archivo
        try:
            model.load_weights(str(self.model_path))
            print("   ✅ Pesos cargados correctamente")
        except Exception as e:
            print(f"   ⚠️  No se pudieron cargar los pesos: {e}")
        
        return model
    
    def extract_embedding(self, audio_path):
        """
        Extrae embedding de audio usando ECAPA-TDNN
        
        Args:
            audio_path: Ruta al archivo de audio
        
        Returns:
            embedding: Vector de 192 dimensiones
        """
        # Cargar audio
        signal, fs = torchaudio.load(str(audio_path))
        
        # Convertir a mono si es estéreo
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        # Resamplear a 16kHz si es necesario
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        
        # Extraer embedding
        with torch.no_grad():
            embedding = self.ecapa.encode_batch(signal)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def predict(self, audio_path):
        """
        Predice la identidad desde un audio
        
        Args:
            audio_path: Ruta al archivo de audio
        
        Returns:
            dict: {
                'identity': str,  # Nombre de la persona o 'unknown'
                'confidence': float,  # Confianza de la predicción
                'probabilities': dict  # Probabilidades por clase
            }
        """
        # Asegurar inicialización
        self.lazy_init()
        
        try:
            # Extraer embedding
            embedding = self.extract_embedding(audio_path)
            embedding = np.expand_dims(embedding, axis=0)  # (1, 192)
            
            # Clasificar
            predictions = self.mlp.predict(embedding, verbose=0)[0]
            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])
            
            # Determinar identidad
            if confidence >= self.confidence_threshold:
                identity = self.label_map[predicted_idx]
            else:
                identity = "unknown"
            
            # Construir resultado
            result = {
                'identity': identity,
                'confidence': confidence,
                'probabilities': {
                    self.label_map[i]: float(predictions[i])
                    for i in range(len(predictions))
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error en predicción de voz: {e}")
            return {
                'identity': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'error': str(e)
            }


# Instancia global del servicio (inicialización perezosa)
voice_service = VoiceRecognitionService()
