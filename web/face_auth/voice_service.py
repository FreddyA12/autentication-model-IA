"""
Voice Recognition Service

Servicio para reconocer identidad por voz usando ECAPA-TDNN (SpeechBrain) + MLP (Keras)
"""

import os
import json
import numpy as np
import torch
import torchaudio
import tensorflow as tf
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
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
            print("[VOICE] Inicializando Voice Recognition Service (ECAPA-TDNN)...")
            
            # Configuración
            self.models_dir = Path(settings.PROJECT_ROOT) / "dataset" / "voice" / "models"
            self.model_path = self.models_dir / "voice_mlp_best.keras"
            self.class_indices_path = self.models_dir / "voice_class_indices.json"
            self.confidence_threshold = getattr(settings, 'VOICE_CONFIDENCE_THRESHOLD', 0.70)
            
            # Verificar que existan los modelos
            if not self.model_path.exists():
                print(f"[WARN] Modelo de voz no encontrado en {self.model_path}")
                self.mlp = None
            else:
                # Cargar MLP
                print("   Cargando MLP...")
                try:
                    self.mlp = tf.keras.models.load_model(str(self.model_path))
                    print(f"   [OK] Modelo MLP cargado exitosamente")
                except Exception as e:
                    print(f"   [WARN] Error al cargar modelo MLP: {e}")
                    self.mlp = None

            # Cargar ECAPA-TDNN
            print("   Cargando ECAPA-TDNN desde SpeechBrain...")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.device = device
                
                from speechbrain.utils.fetching import LocalStrategy
                
                self.encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": device},
                    local_strategy=LocalStrategy.COPY
                )
                print(f"   [OK] ECAPA-TDNN cargado en {device}")
            except Exception as e:
                print(f"[ERROR] Error cargando ECAPA-TDNN: {e}")
                raise
            
            # Cargar mapeo de clases
            print("   Cargando mapeo de clases...")
            if self.class_indices_path.exists():
                with open(self.class_indices_path, 'r', encoding='utf-8') as f:
                    self.label_map = json.load(f)
                    self.label_map = {int(k): v for k, v in self.label_map.items()}
            else:
                print("[WARN] No se encontró el mapeo de clases de voz")
                self.label_map = {}

            print(f"   [OK] Servicio de voz listo ({len(self.label_map)} clases)")
            print(f"   Umbral de confianza: {self.confidence_threshold}")
            
            self._initialized = True
    
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
            embedding = self.encoder.encode_batch(signal)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def predict(self, audio_path):
        """
        Predice la identidad desde un audio
        """
        # Asegurar inicialización
        self.lazy_init()
        
        if self.mlp is None:
            return {
                'identity': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'error': 'Modelo MLP no cargado'
            }
        
        try:
            # Extraer embedding
            embedding = self.extract_embedding(audio_path)
            
            if embedding is None:
                return {
                    'identity': 'error',
                    'confidence': 0.0,
                    'probabilities': {},
                    'error': 'No se pudo extraer embedding'
                }

            # Preparar batch (1, 192)
            embedding_batch = np.expand_dims(embedding, axis=0)
            
            # Clasificar
            predictions = self.mlp.predict(embedding_batch, verbose=0)[0]
            predicted_idx = int(np.argmax(predictions))
            confidence = float(predictions[predicted_idx])
            
            # Determinar identidad
            if confidence >= self.confidence_threshold:
                identity = self.label_map.get(predicted_idx, "Unknown")
            else:
                identity = "unknown"
            
            # Construir resultado
            result = {
                'identity': identity,
                'confidence': confidence,
                'probabilities': {
                    self.label_map.get(i, str(i)): float(predictions[i])
                    for i in range(len(predictions))
                }
            }
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error en predicción de voz: {e}")
            import traceback
            traceback.print_exc()
            return {
                'identity': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'error': str(e)
            }


# Instancia global del servicio (inicialización perezosa)
voice_service = VoiceRecognitionService()
